import numpy as np

class WhatsForLunch:

    def __init__(self, person_matrix, restaurant_matrix):

        self.person_matrix = person_matrix
        self.restaurant_matrix = restaurant_matrix

    def get_person_preference(self, idx):
        person = self.person_matrix[idx]
        person_lc = np.matmul(self.restaurant_matrix, person)
        return person_lc

    def match_preference(self):
        left_reshape = np.swapaxes(self.person_matrix, 0, 1)
        left_reshape.shape, self.restaurant_matrix.shape
        preference_matrix = np.matmul(self.restaurant_matrix, left_reshape)
        return preference_matrix

    def get_best_choice(self, preference_matrix):
        M_rank = preference_matrix.argsort()[::-1] + 1
        np.sum(M_rank, axis=1)
        temp = M_rank.argsort()
        ranks = np.arange(len(M_rank))[temp.argsort()] + 1
        return ranks

    def remove_factor(self, idx):
        person_matrix_removed = np.delete(self.person_matrix, idx, axis=1)
        restaurant_matrix_removed = np.delete(self.restaurant_matrix, idx, axis=1)
        person_matrix_removed.shape, restaurant_matrix_removed.shape
        person_matrix_removed_reshape = np.swapaxes(person_matrix_removed, 0, 1)
        person_matrix_removed_reshape.shape, restaurant_matrix_removed.shape
        removed_M = np.matmul(restaurant_matrix_removed, person_matrix_removed_reshape)
        return removed_M
