U ovom repozitorijumu ima 4 jupyter fajla:

1)Iris.ipynb 

Baza je iz sklearn.datasets, ima 150 podataka o tri vrste cveća: setosa, versicolor i virginica. 
Koristio sam dužinu i širinu latice kao promenljive za predviđanje vrste cveća. Podacima nije bilo potrebno nikakvo dodatno predprocesiranje.
Ovu bazu sam izabrao jer su primeri jasno razdvojivi i hteo sam da vizualizujem kako algoritmi kao sto su SVC, random forest i logisticka regresija donose odluke u zavisnosti od razlicitih vrednosti hiperparametara.
Cilj ovog projekta je bio da shvatim kako 'fino podesiti' model, kao i prepoznam kada overfituje ili underfituje

2) Kredit.ipynb

Baza je takodje iz sklearn.datasets, ima 17000 podataka o kreditnoj sposobnosti klijenata (good / bad). Za predprocesiranje logit modela koristio standardizaciju i vestacko enkodiranje gde se prva kolona svake kategorije izbacuje zbog multikolinearnosti, za SVC standardizaciju i vestacko enkodiranje (bez izbacivanja prve kolone), dok za random forest nisam predprocesuirao podatke.
Fokusirao sam se prvenstveno na vizualizaciju toga kako izgleda f1 statistika na trening i test setu uporedno, kada se menja velicina trening skupa. Koristio sam GridSearchCV da pronadjem najbolji model iz sva tri algoritma i sacuvao ga kao .pkl fajl, kao ciljanu metriku sam postavio scoring='recall', kako bih minimizirao 'lazno pozitivne' klijente. Na test setu se najbolje pokazao RF, pa sam odlucio da udjem u dublju analizu kako RF donosi odluke. Vizualizovao sam intervale vaznosti promenljivih, shap vrednosti, kao i roc auc statistiku kada se menja parametar maksimalne dubine drveta
