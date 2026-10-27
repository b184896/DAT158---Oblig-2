# 1: BESKRIV PROBLEMET

## Omfang / scope

Målet er å utvikle en modell som estimerer en relevant pris for brukte biler basert
på egenskapene i en annonse. Modellen skal senere kunne brukes i en nettside der
en bruker velger egenskaper for en bil og får et prisestimat. Estimatet er ment som
støtte ved kjøp, salg og vurdering av annonser, ikke som en garanti for faktisk
salgspris.

Maskinlæring passer til problemet fordi bilpris ikke bestemmes av én regel. Pris
påvirkes av kombinasjoner av blant annet bilmerke, modell, årsmodell, kilometerstand,
tilstand, drivstoff, girkasse, drivlinje, biltype og geografisk område. Et stort
datasett med tidligere annonser gjør det mulig å lære disse sammenhengene og
generalisere til biler modellen ikke har sett tidligere.

### Målgruppe og behov

Produktet er først og fremst til for:

- privatpersoner som skal kjøpe eller selge bruktbil
- bilforhandlere som vil ha et raskt førsteestimat
- markedsplasser som ønsker prisstøtte eller varsling om uvanlige priser

Løsningen dekker behovet for et raskt og mer datadrevet prisestimat enn en ren
magefølelse eller sammenligning med noen få annonser. Den bør vise at estimatet
er usikkert, siden annonsepris ikke nødvendigvis er det samme som endelig
salgspris.

### Dagens løsninger og manuell løsning

Problemet løses i dag ofte ved å:

1. søke etter lignende biler på Finn, Craigslist eller andre markedsplasser
2. sammenligne årsmodell, kilometerstand, merke, modell og utstyr
3. justere prisen manuelt for tilstand, område og ekstrautstyr
4. eventuelt bruke en verdivurdering fra forhandler, forsikringsselskap eller
   en eksisterende bilkalkulator

En slik metode kan fungere, men er tidkrevende og avhenger av hvilke annonser
personen finner og hvordan de vektes. Tilsvarende løsninger finnes allerede som
prisguider, verdivurderinger og markedsplassfunksjoner. Vår løsning kan
konkurrere ved å være enkel å bruke, vise hvilke opplysninger estimatet bygger
på og være tydelig på usikkerheten.

### System og pipeline

Den planlagte løsningen består av disse delene:

1. innsamling eller import av annonser i `vehicles.csv`
2. datarensing og filtrering
3. feature engineering og encoding
4. modelltrening og validering
5. et API eller en nettside som sender bilens egenskaper til modellen
6. visning av estimat og eventuelt et intervall eller en forklaring

Den samme preprocessing-pipelinen må brukes ved trening og prediksjon. Hvis
datakilden endres, for eksempel ved at kategorinavn eller kolonner endres, kan
det påvirke encoding og dermed modellen. Hvis brukergrensesnittet tillater
verdier som ikke finnes i treningsdataene, må de håndteres som ukjente
kategorier. Nye markedsforhold kan også gjøre at modellen må trenes på nytt.

### Business objective og suksess

Business-målet er å redusere tiden det tar å prise en bil og gjøre prisvurderingen
mer konsistent. For en markedsplass kan dette gi flere relevante annonser,
bedre brukeropplevelse og høyere tillit. For en forhandler kan det redusere
manuelt arbeid og gjøre det raskere å vurdere innbyttebiler.

En første versjon kan regnes som vellykket dersom:

- median absolutt prosentfeil er lav nok til at estimatet er nyttig for brukere
- MAE og RMSE er stabile på validerings- og testdata
- forskjellen mellom train- og testfeil er liten nok til at modellen ikke er
  tydelig overtilpasset
- en bruker kan fylle inn opplysningene og få svar innen omtrent ett sekund
- brukertesting viser at estimatet oppleves forståelig og nyttig

Før produksjon bør business-ytelsen måles med A/B-test eller pilot. Relevante
mål er brukertilfredshet, andel brukere som fullfører en verdivurdering, tid
per vurdering og hvor ofte estimatet ligger innenfor et avtalt prisintervall
fra faktisk salgspris.

### Ressurser

Prosjektet trenger:

- en data scientist/utvikler til datarensing, modellering og evaluering
- en utvikler til API eller nettside dersom modellen skal settes i produksjon
- domeneinnsikt fra en person som kjenner bruktbilmarkedet
- lagring av datasettet og versjoner av modell og preprocessing
- en vanlig utviklingsmaskin for trening på et utvalg, og eventuelt mer CPU/RAM
  ved trening på hele datasettet

Modellen bruker `n_jobs=-1` i Extra Trees og kan derfor bruke flere CPU-kjerner.
Prediksjon krever langt mindre beregningskraft enn trening.

## Metrikker

Dette er et regresjonsproblem. Accuracy er derfor ikke en egnet hovedmetrik.

### Maskinlærings- og software-metrikker

- **MAE (Mean Absolute Error):** gjennomsnittlig absolutt prisfeil i dollar.
  Den er lett å forklare: en MAE på 1 878 betyr i gjennomsnitt omtrent 1 878
  dollar feil på datasettet.
- **RMSE (Root Mean Squared Error):** straffer store feil mer enn MAE og er
  derfor nyttig for å oppdage enkelte svært dårlige prisestimater.
- **MAE som prosent av gjennomsnittspris:** gjør resultatet lettere å
  sammenligne mellom datasett med ulik prisnivå.
- **Median APE (Absolute Percentage Error):** kan brukes som en supplerende
  metrikk fordi prosentfeil er intuitiv for brukeren. Den må tolkes forsiktig
  for svært billige biler.
- **Train/validation/test-gap:** forskjellen mellom treningsfeil og feil på
  data modellen ikke har sett. Et stort gap indikerer overfitting.
- **Latency:** tiden fra brukeren sender bilopplysninger til estimatet vises.
- **Throughput:** hvor mange forespørsler systemet kan håndtere per sekund.

Lav MAE henger direkte sammen med business-målet fordi brukeren ønsker et
prisestimat nær markedsprisen. RMSE er viktig for å sikre at modellen ikke
har enkelte svært misvisende estimater. Latency og throughput sier om modellen
kan brukes i en faktisk nettside.

### Mål for suksess

Det endelige minimumskravet bør fastsettes etter testing mot et representativt
holdout-sett og en brukerpilot. Et foreløpig mål er:

- MAE under 10 % av gjennomsnittsprisen
- liten forskjell mellom validation og test
- median APE som er lav nok til at minst halvparten av estimatene ligger i et
  nyttig intervall for brukeren
- p95-latency under ett sekund for en ferdig lastet modell
- at brukere forstår at resultatet er et estimat og ikke en garantert salgspris

## # 2: DATA

### Datakilde, type og label

Datasettet er `data/vehicles.csv` og består av annonser for brukte kjøretøy.
Dette er tabulære, strukturerte data med både numeriske og kategoriske felt.
Målvariabelen, eller labelen, er `price`. Eksempler på inputvariabler er:

- numeriske: `year`, `odometer` og antall sylindre
- høy-kardinalitets-kategorier: `manufacturer`, `model` og `region`
- andre kategorier: `condition`, `fuel`, `title_status`, `transmission`,
  `drive`, `type` og `state`

I den ferdige modellen utledes også `age` fra bilens årsmodell og året annonsen
ble publisert. `cylinders_num` trekkes ut fra tekstfeltet `cylinders`.

Ved siste kjøring inneholdt råfilen 426 880 rader. Etter manglende verdier,
pris-, års-, kilometer- og aldersfiltre samt fjerning av duplikater ble 359 640
rader brukt i modellen. Fordelingen er 64 % trening, 16 % validering og 20 %
test. Hele testsettet brukes i den endelige diagnostikken.

### Hvordan data kan samles inn

Data kan samles inn fra åpne eller lisensierte annonsekilder, gjennom en avtalt
API, eksport eller periodisk import. En produksjonsløsning må følge vilkårene
til datakilden og ha en plan for oppdatering. Nye data kan brukes til å
oppdage endringer i markedet og til periodisk retrening.

Labelen er prisfeltet i annonsen. Den er relativt enkel å hente, men er ikke
perfekt: annonsepris kan avvike fra faktisk salgspris, og noen annonser kan
ha skrivefeil eller urealistiske priser. Konsistensen forbedres ved å:

- konvertere pris til numerisk verdi
- fjerne åpenbart urealistiske priser
- bruke samme enhet for kilometerstand
- kontrollere gyldige årsmodeller
- fjerne duplikater
- overvåke label-fordelingen over tid

Faktiske salgspriser ville vært en bedre label enn annonsepriser, men er
vanskeligere å samle inn. Dersom de blir tilgjengelige, bør de brukes i en
senere versjon.

### Rensing og feature engineering

Rensekoden i [craigslist_price_model.py](craigslist_price_model.py) gjør blant
annet følgende:

- fjerner rader uten `model`, `year` eller `odometer`
- konverterer pris, årsmodell og kilometerstand til numeriske verdier
- beholder priser fra 500 til 120 000 dollar
- beholder årsmodeller fra 1985 til publiseringsåret + 1
- beholder kilometerstand fra 1 000 til 300 000 miles
- beholder alder fra 0 til 100 år
- gjør manglende kategorier til kategorien `unknown`
- trekker ut et numerisk antall sylindre
- beregner `listing_year`, `listing_month` og `age`
- fjerner annonser som er duplikater basert på stabil annonseidentitet

Numeriske variabler imputeres med median. Høy-kardinalitetsfeltene bruker
target encoding med kryssvalidering, mens øvrige kategoriske felt bruker
one-hot encoding. Dette er viktig fordi vanlig one-hot encoding av `model`
og andre høy-kardinalitetsfelt kan gi svært mange kolonner og gjøre trening
unødvendig tung. Numerisk skalering brukes ikke i Extra Trees, siden
trebaserte modeller ikke trenger skalerte variabler.

### Personvern og etikk

Annonser kan inneholde URL-er, geografiske opplysninger, bilder eller tekst
som indirekte kan knyttes til en person. Slike felt brukes ikke som modellinput
i den endelige modellen. Data bør behandles i tråd med vilkårene for kilden,
GDPR og prinsippet om dataminimering. Rådata bør tilgangsbegrenses og ikke
inneholde unødvendige personopplysninger.

Modellen kan også gi systematiske feil dersom enkelte bilmerker, regioner eller
biltyper er dårlig representert. Resultater bør derfor analyseres separat for
ulike grupper, og brukeren bør få vite at estimatet bygger på historiske
annonsepriser. Det bør være mulig å korrigere eller rapportere åpenbart
urimelige estimater.

# 3: MODELLERING

## Modeller som ble utforsket

I `src/5-modelling/Trying_algorithms.ipynb` ble flere regresjonsmodeller
undersøkt:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- også forsøk med blant annet SVR, AdaBoost og CatBoost finnes i
  utforskningsarbeidet

Modellene ble evaluert med femdelt kryssvalidering på treningsdata, og med
separate train- og validation-resultater. Lineær regresjon fungerte som en
enkel baseline. Et enkelt beslutningstre fikk svært lav treningsfeil, men
viste tydelig overfitting og var derfor ikke en god produksjonskandidat.

XGBoost var den beste kandidaten i den første modellutforskningen. For den
første XGBoost-konfigurasjonen ble følgende målt:

| Modell | CV R2 | CV RMSE | Train R2 | Train RMSE | Validation R2 | Validation RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.5923 | 9 473 | 0.6235 | 9 113 | 0.5315 | 10 507 |
| Decision Tree | 0.4541 | 10 952 | 1.0000 | 14 | 0.4399 | 11 489 |
| Random Forest | 0.7055 | 8 052 | 0.8773 | 5 203 | 0.6790 | 8 697 |
| XGBoost | **0.7434** | **7 509** | 0.8940 | 4 836 | **0.7317** | **7 952** |

Det ble også forsøkt en mer regularisert XGBoost-konfigurasjon med `max_depth=4`,
`reg_alpha=1.0` og `reg_lambda=1.0`. Den ga CV R2 0.7394 og validation RMSE
8 068, og var derfor litt svakere enn den første XGBoost-konfigurasjonen.
Log-transformasjon av målvariabelen ble også prøvd, men ga validation RMSE
8 462.

## Valgt modell og preprocessing

Den endelige modellen i [craigslist_price_model.py](craigslist_price_model.py)
er `ExtraTreesRegressor` med:

- 300 trær
- `min_samples_leaf=3`
- `max_features=0.8`
- `random_state=42`
- alle CPU-kjerner ved trening

Vi startet med XGBoost, men valgte Extra Trees i den endelige pipeline-en fordi
datasettet har høy-kardinalitets-kategorier som `model`, `manufacturer` og
`region`. Target encoding av disse feltene gjør at Extra Trees kan bruke
informasjonen uten et ekstremt stort one-hot-representert feature-rom. Det
gjør treningen enklere og raskere i dette prosjektet, samtidig som modellen
beholder evnen til å lære ikke-lineære sammenhenger.

Hyperparametrene ble justert for å redusere overfitting. `min_samples_leaf=3`
hindrer trærne i å lage blader basert på svært få annonser, og
`max_features=0.8` øker variasjonen mellom trærne.

## Endelig evaluering

Den endelige train/validation/test-diagnostikken bruker modellen som er trent
kun på train-settet. Resultatene er:

| Datasett | Antall rader | MAE | RMSE | MAE % av gjennomsnitt |
|---|---:|---:|---:|---:|
| Train | 230 169 | 456 | 1 578 | 2.4 % |
| Validation | 57 543 | 1 878 | 4 279 | 9.8 % |
| Test | 71 928 | 1 854 | 4 144 | 9.7 % |

Validation- og testresultatene er svært like, noe som tyder på at modellen
generaliserer rimelig stabilt. Train-feilen er likevel betydelig lavere enn
validation- og testfeilen. Det finnes derfor fortsatt noe overfitting, men
gapet er langt mer relevant å vurdere mot unseen-data enn train-resultatet
alene. Testsettet er holdt separat og brukes ikke til å velge hyperparametre.

## Feilanalyse og feature importance

Train-, validation- og testmålingene brukes som en enkel feilanalyse for å
avdekke overfitting og underfitting. Neste steg bør være å undersøke feilene
per bilmerke, biltype, region, prisnivå og årsmodell. Det kan avdekke om
modellen er mindre presis for sjeldne bilmodeller eller dyre biler.

Det finnes en `print_importance`-funksjon i modellfilen som bruker
permutation importance og rapporterer effekten på MAE per original inputkolonne.
Den ble bevisst ikke kjørt i denne gjennomgangen fordi den tok for lang tid.
Når den kjøres på et representativt utvalg, kan resultatene brukes til å:

- kontrollere om modellen bruker fornuftige egenskaper
- oppdage features som ikke bidrar
- vurdere om en feature lekker informasjon
- prioritere bedre datakvalitet for de viktigste feltene

Visualiseringene i [craigslist_visualize.py](craigslist_visualize.py) brukes
til å sammenligne fordelinger og korrelasjoner før og etter filtrering. De kan
avsløre ekstremverdier, skjevheter og endringer som filtreringen introduserer.
Sammen med MAE/RMSE, gruppedelt feilanalyse og feature importance gir dette et
grunnlag for videre forbedring av modellen.
