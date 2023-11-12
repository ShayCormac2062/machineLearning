import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.impute import SimpleImputer


class SVMClassifier:
    def __init__(self):
        self.clf = svm.SVC(kernel='linear', C=1000)

    def train(self, X, y):
        self.clf.fit(X, y)

    def predict(self, point):
        return self.clf.predict(point)

    def get_decision_boundary(self):
        w = self.clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-10, 10)
        yy = a * xx - (self.clf.intercept_[0]) / w[1]
        return xx, yy

    def classify_new_point(self, point):
        return self.predict(point)


def plot_points(X, y):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', label='Class 0', edgecolors='k', facecolor='blue')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='s', label='Class 1', edgecolors='k', facecolor='red')


def plot_decision_boundary(xx, yy):
    plt.plot(xx, yy, 'k-')


def add_new_point(event, classifier):
    if event.button == 2:
        x, y = event.xdata, event.ydata
        new_point = np.array([[x, y]])
        new_class = classifier.classify_new_point(new_point)
        color = 'blue' if new_class == 0 else 'red'
        marker = 'o' if new_class == 0 else 's'
        plt.scatter(x, y, marker=marker, edgecolors='k', facecolor=color, s=200)
        plt.draw()


def main():
    X, y = make_blobs(n_samples=100, centers=2, random_state=6)
    X[0, 0] = np.nan  # Пример добавления NaN

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    classifier = SVMClassifier()

    classifier.train(X, y)

    xx, yy = classifier.get_decision_boundary()

    plot_points(X, y)
    plot_decision_boundary(xx, yy)

    plt.connect('button_press_event', lambda event: add_new_point(event, classifier))

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
