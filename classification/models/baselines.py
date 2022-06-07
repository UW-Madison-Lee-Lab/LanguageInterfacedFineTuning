from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import  BaseEstimator
from collections import Counter
import numpy as np
import torch



def get_optimizer(model, optim):
    if optim == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            momentum=0.9, 
            lr=0.1, 
            weight_decay = 0.0001
            )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=0.001, 
            weight_decay = 0.0001
            )

    return optimizer


class MLPMixupClassifier(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X = None, y = None):
        X_original, y_original = X, y
        num_of_features, num_of_labels = X.shape[1], y.max()+1
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))
        y = F.one_hot(y).float()
        my_dataset = TensorDataset(X, y) # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size = 128) # create your dataloader

        model = MLP(num_of_features, num_of_labels)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, optim='sgd')
        
        loss_sequence = {}
        for epoch in range(15):
            for batch_ndx, sample in enumerate(my_dataloader):
                X, y = sample
                
                lam = get_lambda(alpha = self.alpha)
                X_mix, y_a, y_b = mixup_data(X, y, lam)
                y_hat = model(X_mix)
               
                loss = lam * criterion(y_hat, y_a) + (1 - lam) * criterion(y_hat, y_b)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if epoch % 10 == 0:
                    loss_sequence[epoch] = loss.item()
                    #print('epoch:', epoch, ',loss=', loss.item())

        self.model = model
        self.loss_sequence = loss_sequence
        #print(loss_sequence)

    def predict(self, X = None):
        return self.model(X)

    def score(self, X = None, y = None):
        X = torch.from_numpy(X.astype(np.float32))
        return (np.argmax(self.model(X).detach().numpy(), axis = 1) == y).mean()

    def get_loss_sequence(self):
        return self.loss_sequence

def get_lambda(alpha=1.0):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        if np.random.rand() <= 0.5:
            lam = 1.
        else:
            lam = 0.
    return lam

def mixup_data(x, y, lam):
    batch_size = x.size()[0]
    index = torch.randperm(batch_size) # shuffled index
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b
    

class MLP(torch.nn.Module):
    def __init__(self, num_input_features, num_classes, num_hidden=100):
        super(MLP,self).__init__()
        self.num_hidden = num_hidden
        self.fc1 = torch.nn.Linear(num_input_features, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc3 = torch.nn.Linear(num_hidden, num_classes)

    def forward(self, x, latent=False):
        x = self.fc1(x)
        x = self.fc2(x)
        if latent:
            return x
        y = self.fc3(x)
        return y

class RandomClassifier():
    def fit(self, X, y):
        # check if class is imbalanced
        self.labels = list(set(y))
        self.num_classes = len(self.labels)
        if self.num_classes == 2:
            num_0 = len(np.where(y == self.labels[0]))
            num_1 = len(np.where(y == self.labels[1]))
            if num_0 > 2 * num_1:
                self.major_class = self.labels[0]
            elif num_1 > 2 * num_0:
                self.major_class = self.labels[1]
            else:
                self.major_class = None

    def predict(self, X):
        n = X.shape[0]
        if self.num_classes == 2 and self.major_class is not None:
            return np.asarray([self.major_class] * n)
        else:
            inds = np.random.randint(0, self.num_classes, n)
            return np.asarray([self.labels[i] for i in inds])


class StastisticalClassifier():
    def fit(self, X, y):
        # check if class is imbalanced
        counter = Counter(y)
        self.major_class = counter.most_common(1)[0][0]

    def predict(self, X):
        # return np.asarray([0] * X.shape[0])
        return np.asarray([self.major_class] * X.shape[0])


clf_model = {
    'majorguess': StastisticalClassifier(),
    'svm': SVC(gamma='scale'),
    'logreg': LogisticRegression(random_state=0),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'tree': tree.DecisionTreeClassifier(),
    'nn': MLPClassifier(hidden_layer_sizes=(100, 100), random_state=1, max_iter=300, learning_rate='adaptive', early_stopping=True),
    'xgboost': XGBClassifier(verbosity = 0, silent=True),
    'rf': RandomForestClassifier(max_depth=10, random_state=0),
    'random': RandomClassifier(),
    'nnmixup': MLPMixupClassifier(alpha=1)
}

reg_model = {
    'knn': KNeighborsRegressor(n_neighbors=5),
    'linreg': LinearRegression(),
    'krr': KernelRidge(kernel="rbf", gamma=0.1),
    'nn': MLPRegressor(hidden_layer_sizes=(20, 20), random_state=1, max_iter=300, learning_rate='adaptive', early_stopping=True),
    'xgboost': XGBRegressor(objective='reg:squarederror')
}

clf_model_teachers = {
    'svm': {'poly': make_pipeline(StandardScaler(), SVC(kernel='poly')),'rbf': make_pipeline(StandardScaler(),  SVC(kernel='rbf')), 'sigmoid': make_pipeline(StandardScaler(), SVC(kernel='sigmoid'))},
    'logreg': {'C=1.0':LogisticRegression(random_state=0, C=1.0)},
    'knn': {'K=1':KNeighborsClassifier(n_neighbors=1), 'K=3':KNeighborsClassifier(n_neighbors=3),'K=5': KNeighborsClassifier(n_neighbors=5)},
    'tree': {'d=3':tree.DecisionTreeClassifier(max_depth=3), 'd=5':tree.DecisionTreeClassifier(max_depth=5)},
    'nn': {'width={}'.format(i):MLPClassifier(hidden_layer_sizes=(i, i), random_state=1, max_iter=500) for i in [10,100,200]},
    'xgboost': {'XGB':XGBClassifier(verbosity = 0, silent=True)},
    'rf': {'n_estimators=20':RandomForestClassifier(n_estimators=20, max_depth=10, random_state=0), 'n_estimators=50': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0), 'n_estimators=100' :RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)}
}

param_grids = {
    'rf': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]},
    'svm': {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100]},
    'tree': {'criterion': ("gini", "entropy"), 'max_depth': [3, 5, 20]},
    'nn': {'learning_rate_init': [0.001, 0.01, 0.1]},
    'logreg': {'C': [1, 10, 100]},
    'knn': {'n_neighbors': [5, 1, 3], 'p': [1, 2]},
    'xgboost': {'max_depth': [3, 5, 10]}
}


def predict(clf, x_train, y_train, x_test):
    clf = clf.fit(x_train, y_train)
    return clf.predict(x_test)
