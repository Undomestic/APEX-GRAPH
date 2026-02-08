# 🏁 APEX-GRAPH: F1 Race Strategy Optimizer

A cutting-edge **Formula 1 race strategy optimization system** that uses graph theory, machine learning, and advanced combinatorial algorithms to find the **optimal pit-stop strategy** for any F1 circuit. This platform combines physics-based tire degradation models with multiple optimization techniques to maximize race performance.

## 🎯 Project Overview

APEX-GRAPH is a comprehensive race simulation and optimization platform that answers the critical question in F1 racing: **"When should we pit, and what tire compound should we use?"**

The system analyzes tire degradation patterns, fuel consumption, and race constraints to generate optimal race strategies that minimize total race time while adhering to real-world F1 regulations.

---

## ✨ Key Features

### 🧠 Multiple Optimization Algorithms
- **Graph-Based Optimization**: Dijkstra's shortest path algorithm on custom DAG (Directed Acyclic Graph) representation of race states
- **Bellman-Ford Dynamic Programming**: Optimized DP approach for strategy discovery
- **Monte Carlo Tree Search (MCTS)**: Probabilistic exploration of strategy space
- **Bayesian Optimization**: Hyperparameter tuning for pit-stop timing
- **Game Theory Models**: Multi-agent strategy analysis
- **Integer Linear Programming (ILP)**: Constraint satisfaction optimization
- **Multi-Objective Optimization**: Pareto-optimal strategy generation

### 🛞 Physical Modeling
- **Tire Degradation Models**: ML-powered degradation curves for Soft, Medium, and Hard compounds
- **Ensemble Tire Prediction**: Multiple regression models (exponential decay, polynomial fit, sigmoid drop-off)
- **Fuel Load Impact**: Realistic fuel consumption and weight effects
- **Track Temperature Effects**: Temperature-dependent tire performance variability
- **Tire Cliff Behavior**: Realistic sudden performance drops

### 📊 Strategy Analysis
- **Monte Carlo Testing**: Probability distribution of outcomes
- **Sensitivity Analysis**: How changes in assumptions affect optimal strategy
- **Comparative Analysis**: Compare multiple strategies side-by-side
- **Race Simulation**: Lap-by-lap race execution with realistic constraints

### 🎨 Interactive Visualization
- **Real-time Strategy Visualization**: SVG-based pit-stop timelines
- **3D Track Visualization**: Three.js powered circuit visualization
- **Performance Metrics Dashboard**: Interactive charts and analytics
- **Speedometer Integration**: Real-car style UI elements
- **Strategy Comparison Charts**: Visual strategy comparison

---

## 🏗️ Project Structure

```
.
├── backend/                          # Python + Node.js backend
│   ├── app.py                       # Streamlit web interface
│   ├── server.js                    # Express.js REST API
│   ├── python_bridge.js             # Node.js <-> Python bridge
│   │
│   ├── graph_engine.py              # Custom graph implementation & Dijkstra solver
│   ├── tire_model.py                # ML tire degradation model
│   ├── constraints.py               # F1 racing constraints
│   ├── strategy_analyzer.py         # Strategy analysis & Monte Carlo testing
│   │
│   ├── graph/
│   │   ├── optimizer.py             # ML-powered DP optimizer
│   │   ├── optimizer_improved.py    # Enhanced versions with Bellman DP
│   │   └── optimizer_fallback.py    # Fallback optimization strategies
│   │
│   ├── ml/
│   │   ├── train_model.py          # ML model training
│   │   ├── predict_example.py      # Example predictor
│   │   └── *.pkl                   # Pre-trained models
│   │
│   ├── data/
│   │   └── lap_data.csv            # Historical lap time data
│   │
│   ├── package.json                 # Node.js dependencies
│   └── requirements.txt             # Python dependencies
│
├── backend 1/                        # TypeScript backend (advanced algorithms)
│   ├── src/
│   │   ├── server.ts                # Express TypeScript server
│   │   ├── main.ts                  # Main entry point
│   │   ├── core/
│   │   │   ├── GraphEngine.ts       # TypeScript graph implementation
│   │   │   ├── TireModel.ts         # Ensemble tire model
│   │   │   ├── BellmanDP.ts         # Dynamic programming solver
│   │   │   ├── BayesianOptimizer.ts # Bayesian optimization
│   │   │   ├── MCTSOptimizer.ts     # Monte Carlo Tree Search
│   │   │   ├── GameTheory.ts        # Game theory models
│   │   │   ├── IntegerProgramming.ts# ILP solver
│   │   │   └── MultiObjective.ts    # Pareto optimization
│   │   └── utils/
│   │       └── Visualization.ts     # SVG visualization generation
│   │
│   ├── tsconfig.json
│   └── package.json
│
└── frontend/                         # Web UI
    ├── index.html                   # Main HTML with embedded visualization
    ├── script.js                    # Client-side logic
    ├── server.js                    # Static file server
    ├── styles.css                   # Styling
    └── tools/                       # Debug utilities
```

---

## 🛠️ Technology Stack

### Backend
- **Python**: Streamlit UI, ML models, tire degradation physics
- **Node.js/Express**: REST API server, JavaScript implementations
- **TypeScript**: Advanced algorithm implementations, type safety
- **scikit-learn**: ML model training and prediction
- **pandas**: Data handling and analysis
- **NumPy**: Numerical computations

### Frontend
- **HTML5/CSS3**: Responsive web design
- **JavaScript**: DOM manipulation and API calls
- **Three.js**: 3D track visualization
- **SVG**: Strategy diagrams and charts
- **Chart.js**: Data visualization

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

#### Step 1: Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Step 2: Train ML Models (Optional)
```bash
python ml/train_model.py
```
*Pre-trained models are included; only run this if you want to retrain on new data.*

#### Step 3: Install Node.js Dependencies
```bash
npm install
```

### Frontend Setup
```bash
cd frontend
npm install
```

---

## ▶️ Running the Application

### Option 1: Run Python Streamlit UI (Recommended for Development)
```bash
cd backend
streamlit run app.py
```
This launches an interactive web interface at `http://localhost:8501`

### Option 2: Run Express.js Backend + Static Frontend

**Terminal 1 - Start Backend API:**
```bash
cd backend
npm start
```
Server runs on `http://localhost:5000`

**Terminal 2 - Start Frontend Server:**
```bash
cd frontend
npm start
```
Frontend runs on `http://localhost:8080`

### Option 3: Run TypeScript Backend (Advanced)
```bash
cd backend\ 1
npm run build
npm start
```
Starts on `http://localhost:3000` with advanced algorithm implementations

---

## 📡 API Endpoints

### Health Check
```http
GET /api/health
```
**Response:**
```json
{ "status": "ok" }
```

---

### Optimize Race Strategy
```http
POST /api/optimize
Content-Type: application/json
```

**Request Body:**
```json
{
  "total_laps": 53,
  "soft_count": 5,
  "medium_count": 5,
  "hard_count": 43,
  "track_temp": 25
}
```

**Response:**
```json
{
  "status": "success",
  "optimal": {
    "strategy": [
      "LAP 1: Soft (age 1)",
      "LAP 2: Soft (age 2)",
      "LAP 3: Soft (age 3)",
      "LAP 4: PIT → Medium (age 1)",
      "LAP 5: Medium (age 2)",
      ...
    ],
    "total_time": 4523.45,
    "laps": 53
  },
  "optimizer": "python_ml",
  "runtime_ms": 1234
}
```

---

### Validate Strategy
```http
POST /api/validate_strategy
Content-Type: application/json
```

**Request Body:**
```json
{
  "strategy": ["Soft", "Soft", "Soft", "Medium", "Medium", ...],
  "totalLaps": 53
}
```

**Response:**
```json
{
  "valid": true,
  "issues": [],
  "lapsByCompound": {
    "Soft": 3,
    "Medium": 50
  }
}
```

---

### Recommend Starting Compound
```http
GET /api/recommend_compound?track_type=high_speed&track_temp=25
```

**Response:**
```json
{
  "compound": "Medium",
  "reason": "High-speed track with moderate temperatures"
}
```

---

## 🧮 Optimization Algorithms Explained

### Graph-Based Dijkstra's Algorithm
- **Representation**: Race as a DAG where nodes = (lap, tire, age, fuel)
- **Edges**: Transitions between states with lap time weights
- **Algorithm**: Dijkstra's shortest path finds minimum total race time
- **Complexity**: O(V log V + E) where V = states, E = transitions

### Bellman-Ford Dynamic Programming
- **Approach**: Bottom-up DP computing optimal cumulative time at each state
- **Recurrence**: `DP[lap][tire_age] = min(continue_same_tire, pit_to_fresh_tire) + lap_time`
- **Advantage**: Handles complex constraints better than Dijkstra

### Monte Carlo Tree Search
- **Process**: Builds decision tree through random simulations
- **Selection**: UCB1 algorithm balances exploration vs exploitation
- **Simulation**: Random strategy evaluations from current state
- **Backpropagation**: Updates path values based on outcomes
- **Use Case**: Probabilistic strategy confidence estimation

### Bayesian Optimization
- **Objective**: Find optimal pit timing and compound selection
- **Prior**: Gaussian Process models pit-stop impact
- **Acquisition**: Expected improvement balances exploitation vs exploration
- **Use Case**: Fine-tuning pit-stop windows

### Integer Linear Programming
- **Variables**: Binary pit decisions and compound selection
- **Constraints**: 
  - Pit stop limits per race
  - Tire compound availability
  - Mandatory pit-stop rules
  - Fuel constraints
- **Objective**: Minimize total race time as linear combination of laps and penalties

### Multi-Objective Optimization (Pareto)
- **Objectives**: 
  1. Minimize total race time
  2. Minimize pit stops (reliability)
  3. Minimize tire usage variance
- **Result**: Pareto frontier of non-dominated strategies

---

## 🛞 Physical Model: Tire Degradation

### Mathematical Models Used

#### 1. Exponential Decay Model
```
lap_time(age) = base_time × (1 + degradation_rate)^age
```
**Soft tires**: Fast initial pace, rapid peak and cliff
**Medium tires**: Balanced pace and degradation
**Hard tires**: Slower initial, extended durability

#### 2. Polynomial Fit
```
lap_time(age) = base_time + a×age + b×age²
```
Captures non-linear degradation patterns

#### 3. Sigmoid Drop-off (Cliff Modeling)
```
      1
f(x) = ─────────────────────
       1 + e^(-k(x - x₀))
```
Models sudden performance drops after optimal window

### Tire Properties Database
| Tire | Base Time | Degradation | Peak Laps | Cliff Lap | Range |
|------|-----------|-------------|-----------|-----------|-------|
| Soft | 81.5s | 18%/lap | 8 | 25 | Fast but short-lived |
| Medium | 82.5s | 12%/lap | 12 | 40 | Balanced option |
| Hard | 83.5s | 8%/lap | 15 | 50 | Slower but durable |

---

## 📊 Features in Detail

### Constraint Modeling
- **Pit Stop Rules**: Must pit for different compounds (per F1 2024 regulations)
- **Minimum Tire Age**: Non-degraded tires only
- **Fuel Consumption**: ~1.5kg per lap with weight effects
- **Safety Car Handling**: Potential pit opportunities under SC
- **Track-Specific**: Different base times per circuit
- **Temperature Effects**: Compound performance varies with track temp

### Strategy Analysis
- **Lap-by-Lap Breakdown**: Detailed position for each lap
- **Pit Windows**: Optimal strategic pit timing with variance
- **Tire Management**: Compound preference and aging
- **Fuel Load Profile**: Expected fuel weight through race
- **Risk Assessment**: Sensitivity to fuel consumption assumptions

### Monte Carlo Strategy Testing
- **Iterations**: 10,000+ simulations per strategy
- **Variations**: ±2% lap time randomness (tire temp variance)
- **Distribution**: Mean, std dev, percentiles of total race time
- **Confidence**: 95% confidence intervals on strategy performance
- **Comparison**: Statistical significance testing between strategies

---

## 📈 Performance Metrics

### Optimization Metrics
- **Runtime**: Milliseconds to find optimal strategy
- **Graph Efficiency**: Number of nodes/edges explored
- **Solution Quality**: Gap from theoretical lower bound
- **Memory Usage**: RAM needed for graph construction

### Strategic Metrics
- **Time Saved**: vs. baseline equal-stint strategy
- **Pit Efficiency**: Pit time vs. fresher tire benefit
- **Tire Utilization**: Cost per lap for each compound
- **Fuel Efficiency**: Optimal fuel load strategy

---

## 🧪 Testing & Validation

The project includes:
- **Unit Tests**: `test_bridge.js`, `test_optimize.js`
- **Integration Tests**: Full end-to-end race simulations
- **Model Validation**: Cross-validation with historical F1 data
- **Regression Testing**: Strategy consistency across versions
- **Performance Benchmarks**: Algorithm comparison and profiling

---

## 🔧 Configuration

### Tire Configuration (`tire_model.py`)
Modify `TireProperties` for different compounds or circuits

### Constraint Configuration (`constraints.py`)
Adjust pit-stop rules, tire laps, fuel load per circuit

### Track Configuration (`strategy_analyzer.py`)
Set track-specific base lap times and conditions

### Optimizer Selection (`graph_engine.py`)
Choose between different optimization backends

---

## 📝 Example Usage

### Python (Streamlit)
```python
from tire_model import TireDegradationModel
from graph_engine import StrategyOptimizer
from constraints import RaceConstraints

# Initialize models
tire_model = TireDegradationModel()
constraints = RaceConstraints(total_laps=53, pit_loss_time=20)

# Create optimizer
optimizer = StrategyOptimizer(constraints, tire_model)

# Find optimal strategy
result = optimizer.find_optimal_strategy(
    starting_compound='Medium',
    track_temp=25
)

print(f"Optimal Time: {result['total_time']:.2f}s")
print(f"Strategy: {result['strategy']}")
```

### JavaScript/Express
```javascript
const response = await fetch('/api/optimize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    total_laps: 53,
    soft_count: 5,
    medium_count: 5,
    hard_count: 43,
    track_temp: 25
  })
});

const result = await response.json();
console.log(result.optimal.strategy);
```

### TypeScript
```typescript
import { StrategyOptimizer } from './core/GraphEngine';
import { EnsembleTireModel } from './core/TireModel';

const tireModel = new EnsembleTireModel();
const optimizer = new StrategyOptimizer(53, 20, tireModel);
const result = optimizer.findOptimalStrategy('Medium', 25);

console.log(`Pit Stops: ${result.strategy.length - 1}`);
```

---

## 🎓 Mathematical Foundation

### Problem Formulation
```
Minimize: Σ(lap_time[i]) + pit_penalties
Subject to:
  - Tire compound constraints
  - Pit stop mandatory rules
  - Fuel load limits
  - Track-specific regulations
```

### Graph Representation
```
G = (V, E)
where:
  V = {(lap, tire, age, fuel) | lap ∈ [1,n], tire ∈ {S,M,H}, age ≥ 0}
  E = {(u,v,w) | u→v possible with cost w including pit penalties}
```

### State Space
- **Laps**: 1 to race_length
- **Tire Compounds**: 3 (Soft, Medium, Hard)
- **Tire Age**: 0 to max_laps_per_compound
- **Fuel**: 0 to max_fuel_capacity
- **Total States**: Typically 10,000 - 100,000 depending on circuit

---

## 📚 Documentation Files

- [INTEGRATION_GUIDE.md](backend/INTEGRATION_GUIDE.md) - Backend API integration details
- [INTEGRATION_COMPLETE.md](backend/INTEGRATION_COMPLETE.md) - Python-Node.js bridge documentation

---

## 🤝 Contributing

This project was developed as a optimization challenge. Contributions welcome!

Areas for enhancement:
- Real F1 telemetry data integration
- DRS (Drag Reduction System) effects
- Safety car and yellow flag strategy adaptation
- Multi-car racing dynamics
- Advanced fuel consumption models
- Real-time pit-crew coordination

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🚀 Performance Targets

- **Optimization Speed**: <100ms for 50-lap race
- **Strategy Quality**: 2-5% improvement over baseline
- **Monte Carlo Accuracy**: <1% variance in recommendations
- **Graph Construction**: <500ms for 100,000+ node graphs
- **Memory Footprint**: <100MB for full race simulation

---

## 🤖 AI/ML Components

- **Tire Degradation Prediction**: scikit-learn RandomForestRegressor
- **Feature Engineering**: Age² polynomial, temperature interaction terms
- **Model Training**: Cross-validation on historical lap data
- **Ensemble Methods**: Combines multiple prediction models
- **Hyperparameter Tuning**: Grid search optimization with Bayesian backup

---

## 📞 Support & Questions

For questions about specific algorithms, optimization approaches, or integration help:
1. Check the INTEGRATION_GUIDE.md in backend/
2. Review example files in ml/ and graph/ directories
3. Examine test files for usage patterns

---

**APEX-GRAPH**: Where mathematics meets motorsports.
**Optimize your race. Dominate the podium.** 🏆

---

*Last Updated: February 2026*
*Version: 1.0.0*
