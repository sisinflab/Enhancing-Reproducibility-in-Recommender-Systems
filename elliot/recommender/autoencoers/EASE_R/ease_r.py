"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import time
from operator import itemgetter

import numpy as np
import scipy
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import tqdm
import torch

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

from similaripy import similarity


class EASER(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_neighborhood", "neighborhood", "neighborhood", -1, int, None),
            ("_l2_norm", "l2_norm", "l2_norm", 1e3, float, None)
        ]

        self.autoset_params()
        if self._neighborhood == -1:
            self._neighborhood = self._data.num_items

    @property
    def name(self):
        return f"EASER_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k):
        # return {u: self.get_user_predictions(u, mask, k) for u in self._data.train_dict.keys()}
        recs = {}
        for i in tqdm(range(0, len(self._data.train_dict.keys()), 1024), desc="Processing batches",
                      total=len(self._data.train_dict.keys()) // 1024 + (1 if len(self._data.train_dict.keys()) % 1024 != 0 else 0)):
            batch = list(self._data.train_dict.keys())[i:i + 1024]
            mat = self.get_user_recs_batch(batch, mask, k)
            proc_batch = dict(zip(batch, mat))
            recs.update(proc_batch)
        return recs

    def get_user_predictions(self, user_id, mask, top_k=10):
        user_id = self._data.public_users.get(user_id)
        b = self._preds[user_id]
        a = mask[user_id]
        b[~a] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(b.data)])

        indices = np.array(indices)
        values = np.array(values)
        local_k = min(top_k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    def get_user_recs_batch(self, u, mask, k):
        u_index = itemgetter(*u)(self._data.public_users)
        users_recs = np.where(mask[u_index, :], self._preds[u_index, :].toarray(), -np.inf)
        index_ordered = np.argpartition(users_recs, -k, axis=1)[:, -k:]
        value_ordered = np.take_along_axis(users_recs, index_ordered, axis=1)
        local_top_k = np.take_along_axis(index_ordered, value_ordered.argsort(axis=1)[:, ::-1], axis=1)
        value_sorted = np.take_along_axis(users_recs, local_top_k, axis=1)
        mapper = np.vectorize(self._data.private_items.get)
        return [[*zip(item, val)] for item, val in zip(mapper(local_top_k), value_sorted)]

    def to_scipy_csr(self, sparse_tensor):
        """Converts a PyTorch sparse tensor to a SciPy CSR matrix."""
        if not sparse_tensor.is_coalesced():
            sparse_tensor = sparse_tensor.coalesce()  # Ensure COO format is coalesced

        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        shape = sparse_tensor.shape

        return scipy.sparse.csr_matrix((values.numpy(), (indices[0].numpy(), indices[1].numpy())), shape=shape)

    def train(self):
        if self._restore:
            return self.restore_weights()


        start = time.time()

        self._train = self._data.sp_i_train_ratings

        # self._similarity_matrix = safe_sparse_dot(self._train.T, self._train, dense_output=False)
        self._similarity_matrix = similarity.dot_product(self._train.T, self._train,
                                                         k=self._train.shape[0], format_output= 'csr')


        diagonal_indices = np.diag_indices(self._similarity_matrix.shape[0])
        item_popularity = np.ediff1d(self._train.tocsc().indptr)
        self._similarity_matrix.setdiag(item_popularity + self._l2_norm)

        if torch.cuda.is_available():
            self.logger.info(f"Use CUDA for Inverse")
            self._similarity_matrix = torch.tensor(data=self._similarity_matrix.todense(),
                                                   dtype=torch.float32).cuda()
            torch.cuda.synchronize()
            P = torch.linalg.inv(self._similarity_matrix).cpu().numpy()
            torch.cuda.empty_cache()
        else:
            self.logger.info(f"Classical Inverse")
            P = np.linalg.inv(self._similarity_matrix.todense())

        self._similarity_matrix = P / (-np.diag(P))

        self._similarity_matrix[diagonal_indices] = 0.0

        end = time.time()
        self.logger.info(f"The similarity computation has taken: {end - start}")

        if torch.cuda.is_available():
            sparse_train = torch.sparse_coo_tensor(self._train.nonzero(), self._train.data, self._train.shape).cuda()
            self._similarity_matrix = torch.tensor(data=self._similarity_matrix,
                                                   dtype=torch.float32).cuda()
            self._preds = torch.sparse.mm(sparse_train, self._similarity_matrix).cpu().to_sparse_coo()
            self._preds = self.to_scipy_csr(self._preds)
            self.logger.info(f"{type(self._preds)}")
            torch.cuda.empty_cache()
        else:
            self._preds = safe_sparse_dot(self._train, scipy.sparse.csr_matrix(self._similarity_matrix)).tocsr()

        self.evaluate()
