import numpy as np

class Logit():
    
    def __init__(self, lr=0.1, regularizacija='', reg=0.01, maxIter=None, nIter = None):      
        '''Za regularizaciju uneti string 'l1' ili 'l2'
        reg je regularizacioni parametar lambda 
        learning rate: pocetna brzina ucenja (polovi se na svakih 1000 iteracija) '''
        
        self.lr = lr
        self.regularizacija = regularizacija
        self.reg = reg
        self.maxIter = maxIter
        self.preciznostTrain= None
        self.preciznostDev = None
        self.historyW = []


    def sigmoid(self, x, w = None):
        if w is None:
            w = self.w
        z = x @ w
        z = np.clip(z, -500, 500)
        p = 1 / (1 + np.exp(-z))
        return p

    def softmax(self,x, w = None):
        if w is None:
            w = self.w
        z = x @ w
        z -= np.max(z, axis = 1, keepdims = True)
        p = np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)
        return p
    
    def predict(self, x, y =None):
        """Racuna binarne predikcije za ulazne podatke x.
        Ako je prosleđen y, vraća dvojku (preciznost, predikcije)."""

        if self.aktivacija == 'sigmoid':
            p = self.sigmoid(x)
            pred = (p > 0.5).astype(int)
            if y is not None:
                return (np.mean(pred == y), pred)
            else:
                return pred

        elif self.aktivacija == 'softmax':
            p = self.softmax(x)
            pred = np.argmax(p, axis=1)
            if y is not None:
                return (np.mean(pred == np.argmax(y, axis=1)), pred)
            else:
                return pred
        
    def predictProba(self,x):
        """Vraca predikcije verovatnoca za ulazne podatke x. """
        
        if self.aktivacija == 'sigmoid':
            return self.sigmoid(x)
            
        elif self.aktivacija == 'softmax':
            return self.softmax(x)
        
    def fit(self, x, y, xdev, ydev, randomState = 42, aktivacija = 'sigmoid'):
        """Treniranje modela koristeći grupni gradijentni spust (batch gradient descent).
        Funkcija prati preciznost na trening i dev skupu.
        Svakih 100 iteracija se čuvaju trenutne težine u self.historyW.
        Ako preciznost na dev skupu opadne u odnosu na 
        4 evaluacije unazad, smatra se da je dostigao plato i
        model se vraca na težine iz te iteracije i vraca
        dvojku (preciznost na trening skupu, preciznost na dev skupu)."""
        
        m, n = x.shape
        np.random.seed(randomState)
        self.w = np.random.rand(n,y.shape[1]) - .5
        trainscore = []
        devscore = [0 for _ in range(4)]
        i = 0
        lr = self.lr
        epsilon=.0000001
        self.aktivacija = aktivacija
        
        while True:
            z = x @ self.w
            
            if self.aktivacija == 'sigmoid':
                p = self.sigmoid(x)
                gradijenti = (x.T @ (p - y)) / m
                l = -np.mean(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))                
                
            elif self.aktivacija == 'softmax':
                z -= np.max(z, axis = 1, keepdims = True)
                p = np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)
                gradijenti = (x.T @ (p - y)) / m
                l = -np.mean(np.sum(y * np.log(p + epsilon), axis=1))
                
            else:
                raise ValueError("Aktivacija mora biti 'sigmoid' ili 'softmax'")

            
            if self.regularizacija.lower() == 'l1':
                l += (self.reg / m) * np.sum(np.abs(self.w[1:]))
                gradijenti[1:] += (self.reg / m) * np.sign(self.w[1:])
            elif self.regularizacija.lower() == 'l2':
                l += (self.reg / (2 * m)) * np.sum(np.square(self.w[1:]))
                gradijenti[1:] += (self.reg / m) * self.w[1:]

            self.w -= lr * gradijenti
            
            if i % 100 == 0:
                self.historyW.append(self.w.copy())
                preciznostTrain, _ = self.predict(x,y)
                trainscore.append(preciznostTrain)
                
                preciznostDev, _ = self.predict(xdev, ydev)
                devscore.append(preciznostDev)
            
            if i > 300 and (devscore[-1] - devscore[-5]) <= 0:
                print(f"Optimalni parametri su iz {i-300} iteracije")
                self.w = self.historyW[-4]
                break
            
            if self.maxIter is not None and i >= self.maxIter:
                print(f"Maksimalan broj iteracija ({self.maxIter}) dostignut.")
                break
            
            if i % 1000 == 0 and i > 0:
                lr *= 0.5
                print(f"Learning rate: {lr}, iteracija {i}")
                

            i += 1
        self.nIter = i - 300
        self.preciznostTrain, _ = self.predict(x,y)
        print("Preciznost na trening setu:", self.preciznostTrain)

        self.preciznostDev, _ = self.predict(xdev,ydev)
        print("Preciznost na dev setu:", self.preciznostDev)
        
        return self.preciznostTrain, self.preciznostDev

    
    def fitReg(self, x, y, xdev, ydev, listaRegularizacije):
        '''Fituje model za svaki parametar lambda iz liste, cuva rezultate na dev skupu
        na kraju fituje model sa lokalno optimalnim lambda parametrom i vraca recnik
        {lambda : rezultat na dev skupu}'''

        rezultati = []
        for i in range(len(listaRegularizacije)):
            self.reg = listaRegularizacije[i]
            _, devScore = self.fit(x, y, xdev, ydev)
            rezultati.append(devScore)
            
        self.w = self.historyW[np.argmax(rezultati)]
        self.reg = listaRegularizacije[np.argmax(rezultati)]
        self.fit(x, y, xdev, ydev)
        return {self.reg : rezultati[np.argmax(rezultati)]}