import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов нужно брать среднее двух соседних при сортировке значений признака
    * Поведение функции в случае константного признака может быть любым
    * При одинаковых приростах критерия Джини для нескольких порогов нужно выбирать сплит, у которого значение порога минимально
    * Достаточно поддерживать только бинарную классификацию.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на две различные подвыборки или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds, len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # удаляем nan-ы
    mask = ~np.isnan(feature_vector)

    feature_vector = feature_vector[mask]
    target_vector = target_vector[mask]

    # преобразуем классы в их номера для удобства
    classes, R_count = np.unique(target_vector, return_counts=True)
    target_vector = np.searchsorted(classes, target_vector)

    # сортируем по значениям признака
    inds = np.argsort(feature_vector)
    feature_vector = feature_vector[inds]
    target_vector = target_vector[inds]

    # создаем смещенный вектор
    feature_vector_rolled = np.roll(feature_vector, -1)
    feature_vector_rolled[-1] = feature_vector[-1]

    # маска - меняется ли значение при переходе на след. индекс
    threshold_mask = feature_vector_rolled != feature_vector

    # все значения thresholds
    thresholds = (feature_vector + feature_vector_rolled) / 2

    # считаем для каждой ячейки кол-во положительных эл-тов в L и R, если все до этой ячейки включительно относить в L
    L_pos_counts = np.cumsum(target_vector)
    total_pos_counts = np.sum(target_vector)
    R_pos_counts = total_pos_counts - L_pos_counts

    L_total_counts = np.arange(1, len(feature_vector) + 1)
    R_total_counts = np.maximum(np.arange(len(feature_vector) - 1, -1, step=-1), 1)

    # считаем доли положительных классов для каждого разбиения
    L_pos_share = L_pos_counts / L_total_counts
    R_pos_share = R_pos_counts / R_total_counts

    # считаем отдельные и общий gini для каждого разбиения
    L_gini = 1 - (L_pos_share) ** 2 - (1 - L_pos_share) ** 2
    R_gini = 1 - (R_pos_share) ** 2 - (1 - R_pos_share) ** 2

    ginis = -L_total_counts / len(feature_vector) * L_gini - R_total_counts / len(feature_vector) * R_gini

    # берем thresholds и gini только в точках роста признака
    ginis, thresholds = ginis[threshold_mask], thresholds[threshold_mask]

    best_ind = np.argmax(ginis)

    return thresholds, ginis, thresholds[best_ind], ginis[best_ind]


class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже могут быть намеренно оставлены некоторые ошибки.
    Их нужно исправить в рамках задания.
    """
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {
            'depth': 0,
        }
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None and node['depth'] >= self._max_depth:
            node["type"] = "terminal"
            # https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-array
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            # https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-array
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            if np.all(sub_X[:, feature] == sub_X[0, feature]):
                continue
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature]) 
                ratio = {}
                for key, current_count in counts.items():
                    current_click = 0
                    if key in clicks:
                        current_click = clicks[key]
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)

            if self._min_samples_leaf is not None:
                sorted_vals = np.sort(feature_vector)
                sorted_vals_rolled = np.roll(sorted_vals, -1)
                sorted_vals_rolled[-1] = sorted_vals[-1]
    
                threshold_mask = sorted_vals != sorted_vals_rolled
    
                L_counts = np.arange(1, len(threshold_mask) + 1)[threshold_mask]
                total_count = len(sorted_vals)
                good_thresholds = (L_counts >= self._min_samples_leaf) & ((total_count - L_counts) >= self._min_samples_leaf)

                if sum(good_thresholds) == 0:
                    continue

                thresholds, ginis = thresholds[good_thresholds], ginis[good_thresholds]
                best_ind = np.argmax(ginis)
                gini = ginis[best_ind]
                threshold = thresholds[best_ind]
            
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        
        node["left_child"], node["right_child"] = {
            'depth': node['depth'] + 1,
        }, {
            'depth': node['depth'] + 1,
        }
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']

        if self._feature_types[node['feature_split']] == 'real':
            if x[node['feature_split']] < node['threshold']:
                return self._predict_node(x, node['left_child'])
        elif x[node['feature_split']] in node['categories_split']:
            return self._predict_node(x, node['left_child'])
        return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError('Binary classification only')
        self._classes = classes
        y = (y == classes[1]).astype(int)
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return self._classes[np.array(predicted, dtype=int)]
