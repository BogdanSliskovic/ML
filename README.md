U ovom repozitorijumu ima 4 jupyter fajla:

1)Iris.ipynb

Baza je iz sklearn.datasets, ima 150 podataka o tri vrste cveća: setosa, versicolor i virginica.
Koristio sam dužinu i širinu latice kao promenljive za predviđanje vrste cveća. Podacima nije bilo potrebno nikakvo dodatno predprocesiranje.
Ovu bazu sam izabrao jer su primeri jasno razdvojivi i hteo sam da vizualizujem kako algoritmi kao sto su SVC, random forest i logisticka regresija donose odluke u zavisnosti od razlicitih vrednosti hiperparametara.
Cilj ovog projekta je bio da shvatim kako 'fino podesiti' model, kao i prepoznam kada overfituje ili underfituje

2) Kredit.ipynb

Baza je takodje iz sklearn.datasets, ima 17000 podataka o kreditnoj sposobnosti klijenata (good / bad). Za predprocesiranje logit modela koristio sam standardizaciju i vestacko enkodiranje gde se prva kolona svake kategorije izbacuje zbog multikolinearnosti, za SVC standardizaciju i vestacko enkodiranje (bez izbacivanja prve kolone), dok za random forest nisam predprocesuirao podatke.
Fokusirao sam se na vizualizaciju f1 statistike na trening i test setu uporedno, kada se menja velicina trening skupa i hiperparametri, kako bih jasnije uocio kako izgleda overfitting  i underfitting i kako da podesim hiperparametre da to izbegnem. Koristio sam GridSearchCV kako bih pronašao optimalne hiperparametre za sva tri algoritma i sacuvao ga u .pkl formatu. Kao ciljanu metriku sam postavio scoring='recall', kako bih minimizovao lazno pozitivne predikcije. Na test setu se najbolje pokazao RF, pa sam se detaljnije posvetio analizi njegovog ponašanja. Vizualizovao sam intervale vaznosti promenljivih (feature importance), SHAP vrednosti, kao i ROC AUC pri razlicitim vrednostima maksimalne dubine drveta. Za kraj, izdvojio sam podatke iz test skupa gde RF model pravi greške

3) Titanik.ipynb

Baza je iz sns.load_dataset, ima podatke o 900 putnika na titaniku sa njihovim karakteristikam (starost, pol, klasa, broj članova porodice, tarifa karte, luka ukrcavanja) i ciljanu promenjivom da li su preziveli ili nisu.
Prvo sam uklonio par kolona koje ne dodaju dodatne informacije i proverio deskriptivne statistike, gde sam video da varijabla koja oznacava palubu na koju su se putnici ukrcali ima samo 200 non - null vrednosti. Odlucio sam da nedostajuce vrednosti imputiram sa KNNimputer - om. Nakon toga napravio sam nove promenjive koje obelezavaju velicinu porodice i tacnije mesto ukrcavanja (kombinacija klase i palube). Kada sam delio podatke na train i test, obratio sam paznju na stratifikaciju uzorka (zbog nebalansiranosti klasa - samo 38 posto ljudi je prezivelo). 
Sa RandomSearchCV trenirao sam SVC, logit, random forest i ridge classifier sa raznim setovima hiperparametara i rezultate kao sto su train score, test score, f1, ROC AUC sam sacuvao u promenjivoj 'rezultati' koju sam sacuvao u .pkl formatu kao i po jedan od najboljih seta hiperparametara za svaki model. 
Kako se SVC model najbolje pokazao, detaljnije sam se posvetio analizi njegovog ponasanja. Analizom glavnih komponenti (PCA) sam smanjio dimenzije prvo na dve a zatim na tri, i vizualizovao granice donosenja odluka, ako i matricu zabune (confusion matrix) i ROC AUC na trening i na test podacima.
Nakon toga opet sam ucitao po jedan od modela sa najboljim hiperparametrima i napravio VotingClassifier koristeci soft i hard glasanje. Model sa soft glasanjem se pokazao najboljim pa sam opet smanji dimenzije sa PCA na dve odnosno tri i uradio vizualizaciju donosenja odluka. 

Nakon toga hteo sam da vidim da li klasterizacija podataka moze pomoci modelu. Napravio sam pipeline koji radi predprocesiranje za SVC model (nisam mogao za soft voting jer bih morao ponovo da pravim predproc za sve modele i opet da ih fitujem), sa druge strane radi GridSearch od svakog neparnog broja izmedju 3 i 59 za najbolji hiperparametar broja klastera koji algoritam KMeans trazi u podacima (za njegovo predprocesuiranje radi samo standardizaciju), onda dodaje kolonu koja oznacava broj klastera i fituje SVC model. GridSearch je pronasao n_clusters=29 kao optimalni hiperparametar koji je na trening setu identican kao i obican SVC - a, ali na test setu ima dosta losije rezultate. Kako bih smanjio overfitting, smanjio sam grid na neparne brojeve izmedju 3 i 21 (nije pomoglo).
Kako KMeans ocigledno ne pomaze probao sam da odradim klasterizaciju sa DBSCAN - om (ne treba predefinisan broj klastera i robustniji je od KMeans na klastere razlicitih oblika). Da bih dobio ideju koja vrednost epsilona je prikladna ovom skupu podataka pomocu KNN algoritma vizualizovao sam udaljenost do pete najblize tacke, primetio sam da ima dva skoka za epsilon vrednosti 1 i 2, dosao sam do zakljucka da je vrednost 1.65 zadovoljavajuca za epsilon (otkriva podklastere i nema previse netipicnih vrednosti), ima 6 klastera i 69 netipicnih vrednosti. Dodao sam kolonu koja obelezava kom klasteru pripada svaki putnik u trening setu i ponovo trenirao model.


4) MNIST.ipynb
