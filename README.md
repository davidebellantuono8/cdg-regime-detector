# CDG Macro Regime Detector v2.5 — Decision Engine con Market Proximity

Applicazione Streamlit unica con tre blocchi integrati:

1. **Market Regime Engine** — determina sempre il regime operativo corrente usando gli 8 basket Equal Weight.
2. **Macro Leading Engine 2–3M** — anticipatore macro sui medesimi 8 regimi.
3. **Decision Engine V1.1** — combina Market Current, Market Emerging e Macro Next senza mai lasciare il portafoglio senza regime.

Gli 8 regimi sono first-class: Recession, Debasement, Stagflation, Reflation, Dollar Weakness, Goldilocks Economy, Disinflation / Soft Landing, Deflation.

## Novità v2.5

La v2.5 **non modifica Market Engine, 39 serie FRED, Macro Engine 12D, archetipi, pesi 70/30 o walk-forward**. Cambia soltanto il Decision Engine aggiungendo il **Market Proximity Filter**.

L'obiettivo è evitare che Macro + Emerging anticipino un regime ancora troppo lontano dalla leadership Current del mercato.

### Early switch normale — 2 mesi

Richiede contemporaneamente:

- stesso Macro Next per 2 mesi;
- Macro Next = Market Emerging;
- Market Emerging >= 60;
- Macro Confidence >= MEDIUM-HIGH;
- Macro Agreement >= MEDIUM;
- Current del candidato >= 60;
- **candidato Top-3 Current oppure entro 10 punti dal Market Leader**.

### Quick Early Switch — 1 mese

Caso eccezionale. Richiede:

- Macro Next = Market Emerging;
- Macro Confidence = HIGH;
- Macro Agreement = HIGH;
- Macro Score >= 75;
- Emerging >= 70;
- Current candidato >= 60;
- **candidato Top-2 Current oppure entro 7 punti dal Market Leader**.

La v2.5 mostra inoltre in Dashboard/Decision Engine:

- Current del candidato Macro;
- rank Current del candidato;
- gap in punti dal Market Leader;
- esito dei filtri di prossimità nello storico esportato.

## Market Regime Engine

Metodologia invariata:

- Current = 45% Forza + 30% Persistenza + 20% Relative Performance + 5% Accelerazione
- Emerging = 25% Forza + 10% Persistenza + 25% Relative Performance + 40% Accelerazione
- switch normale Market: Current nuovo leader >= 60, vantaggio >= 7 punti, 2 conferme mensili
- quick switch Market: Current >= 75, Accelerazione >= 70, Persistenza >= 60
- il regime operativo è **sempre presente**
- stato operativo: STABLE / WATCH / TRANSITION

## Macro Leading Engine 2–3M

Invariato rispetto alla v2.3/v2.4:

- 39 serie FRED automatiche;
- 6 fattori: US Growth, Global Growth, Inflation, Financial Stress, Liquidity, USD;
- 12 dimensioni: Level e Impulse separati per ciascun fattore;
- Economic Score 70%;
- Empirical Score 30% con Ridge walk-forward sui basket Equal Weight;
- Macro Regime Score 0–100;
- confidence dal gap fra primo e secondo Macro Score;
- validazione walk-forward 1M/2M/3M integrata.

Parametri empirici fissati ex ante:

- Ridge lambda = 8
- minimo training = 36 mesi
- peso empirico = 30%
- smoothing finale EWM alpha = 0,55, solo backward-looking

## Decision Engine V1.1

Azioni possibili:

- **HOLD** — regime operativo confermato o early regime già assunto e ancora supportato.
- **WATCH** — divergenza Macro/Market con segnale non sufficiente.
- **PREPARE** — convergenza interessante ma manca una conferma piena o la Market Proximity è insufficiente.
- **SWITCH** — nuovo regime anticipato solo con tutte le condizioni previste.

Un early regime è sticky: servono 2 mesi consecutivi di perdita delle conferme prima di tornare al Market Regime validato.

## Dati macro FRED

La key FRED può essere salvata localmente in:

`config/fred_api_key.txt`

Non viene esportata. Download paralleli e cache incrementale restano attivi.

## Avvio Windows

1. Estrai completamente lo ZIP in una cartella nuova.
2. Doppio clic su `AVVIA_APP.bat`.
3. Inserisci/salva la FRED API key se non è già configurata.
4. Carica il CSV/XLSX con le 8 serie prezzi dei basket Equal Weight.
5. L'app calcola Market + Macro + Decision Engine.

## Export

`detector_history.xlsx` contiene:

- Market Regime History
- Market Active Path
- Macro Regime History
- Macro Factors
- Macro Validation
- Empirical Predictions
- Decision History
- Decision Validation
- FRED Status

Nel foglio **Decision History** la v2.5 aggiunge anche:

- MacroCandidateCurrent
- MacroCandidateCurrentRank
- MacroCandidateGapToLeader
- NormalProximityOK
- QuickProximityOK

## Nota sul backtest macro

Le serie mensili sono spostate conservativamente di un mese per ridurre il look-ahead da data di riferimento/pubblicazione. La calibrazione sui basket è walk-forward. Lo storico macro non è ancora completamente vintage-safe: una validazione definitiva delle revisioni richiederà ALFRED/vintage.
