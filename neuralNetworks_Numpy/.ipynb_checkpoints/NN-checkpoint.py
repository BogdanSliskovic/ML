{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7ccd61eb-d42c-44c7-a8fa-34420416db53",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'Logit' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mclass\u001b[39;00m \u001b[38;5;21;01mNeuralNetwork\u001b[39;00m(Logit):\n\u001b[0;32m      2\u001b[0m     \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__init__\u001b[39m(\u001b[38;5;28mself\u001b[39m, dimenzijeSlojeva, lr\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.1\u001b[39m, regularizacija\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m'\u001b[39m, reg\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.01\u001b[39m, maxIter\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, seed\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m42\u001b[39m):\n\u001b[0;32m      3\u001b[0m         \u001b[38;5;28msuper\u001b[39m()\u001b[38;5;241m.\u001b[39m\u001b[38;5;21m__init__\u001b[39m(lr, regularizacija, reg, maxIter)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'Logit' is not defined"
     ]
    }
   ],
   "source": [
    "class NeuralNetwork(Logit):\n",
    "    def __init__(self, dimenzijeSlojeva, lr=0.1, regularizacija='', reg=0.01, maxIter=None, seed=42):\n",
    "        super().__init__(lr, regularizacija, reg, maxIter)\n",
    "        self.aktivacija = 'sigmoid'\n",
    "        np.random.seed(seed)\n",
    "        self.weights = [np.random.rand(dimenzijeSlojeva[i], dimenzijeSlojeva[i+1]) - 0.5 for i in range(len(dimenzijeSlojeva)-1)]\n",
    "\n",
    "    def forward(self, x, y, loss = True):\n",
    "        aktivacija = x\n",
    "        for i, w in enumerate(self.weights):\n",
    "            if i == len(self.weights) - 1:\n",
    "                aktivacija = self.softmax(aktivacija, w)\n",
    "            else:\n",
    "                aktivacija = self.sigmoid(aktivacija, w)\n",
    "\n",
    "        if loss is not True:\n",
    "            return aktivacija\n",
    "        else:\n",
    "            loss = -np.sum(y * np.log(aktivacija + 1e-6)) / len(x)\n",
    "        \n",
    "            if self.regularizacija.lower() == 'l2':\n",
    "                loss += (self.reg / (2 * len(x))) * sum([np.sum(w ** 2) for w in self.weights])\n",
    "        \n",
    "            elif self.regularizacija.lower() == 'l1':\n",
    "                loss += (self.reg / len(x)) * sum([np.sum(np.abs(w)) for w in self.weights])\n",
    "        \n",
    "            return loss\n",
    "\n",
    "\n",
    "    def gradijenti(self, x, y, epsilon=.00001):\n",
    "        grads = [np.zeros_like(w) for w in self.weights]\n",
    "\n",
    "        for sloj, w in enumerate(self.weights):\n",
    "            g = grads[sloj]\n",
    "\n",
    "            for i in range(w.shape[0]):\n",
    "                for j in range(w.shape[1]):\n",
    "                    original = w[i, j]\n",
    "\n",
    "                    w[i, j] = original + epsilon\n",
    "                    fPlus = self.forward(x, y)\n",
    "\n",
    "                    w[i, j] = original - epsilon\n",
    "                    fMinus = self.forward(x, y)\n",
    "\n",
    "                    g[i, j] = (fPlus - fMinus) / (2 * epsilon)\n",
    "                    w[i, j] = original\n",
    "        return grads\n",
    "        \n",
    "    def trainStep(self, x, y, epsilon=.00001, learning_rate=0.01):\n",
    "        loss = self.forward(x, y)\n",
    "        grads = self.gradijenti(x, y, epsilon)\n",
    "        self.weights = [w - learning_rate * g for w, g in zip(self.weights, grads)]\n",
    "        return loss\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "77f5ea95-4758-49a4-8fb5-fdb259f119d3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
