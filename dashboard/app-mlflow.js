// --- Explore State ---
const ExploreState = {
    experiments: [],
    selectedExpId: null,
    runs: [],
    selectedRunIds: new Set(),
    sortMetric: 'accuracy',
    compareChart: null,
    artifactRunId: null,
};

// --- App State ---
const State = {
    currentView: 'overview',
    selectedModel: 'Random Forest',
    isTraining: false,
    mlflowData: {
        experiments: [],
        runs: [],
        summary: null,
        selectedExperiment: null,
        loading: false
    },
    history: [
        { id: 'EXP-001', model: 'Logistic Regression', accuracy: '84.2%', f1: '0.78', status: 'Completed', date: '2026-03-01 09:12' },
        { id: 'EXP-002', model: 'Random Forest', accuracy: '89.5%', f1: '0.86', status: 'Completed', date: '2026-03-01 10:45' },
    ],
    models: {
        'Logistic Regression': {
            desc: 'Linear model for binary classification. Efficient and interpretable.',
            params: [
                { key: 'penalty', name: 'Penalty', type: 'select', options: ['l2', 'l1', 'elasticnet'], value: 'l2' },
                { key: 'C', name: 'C (Inverse Regularization)', type: 'number', value: 1.0, step: 0.1 },
                { key: 'solver', name: 'Solver', type: 'select', options: ['lbfgs', 'liblinear', 'saga'], value: 'lbfgs' }
            ]
        },
        'Random Forest': {
            desc: 'Ensemble of decision trees. Robust to noise and non-linear patterns.',
            params: [
                { key: 'n_estimators', name: 'N Estimators', type: 'number', value: 100, step: 10 },
                { key: 'max_depth', name: 'Max Depth', type: 'number', value: 12, step: 1 },
                { key: 'min_samples_split', name: 'Min Samples Split', type: 'number', value: 2, step: 1 }
            ]
        },
        'XGBoost': {
            desc: 'Gradient Boosted Decision Trees. High performance, used for structured data.',
            params: [
                { key: 'n_estimators', name: 'N Estimators', type: 'number', value: 200, step: 50 },
                { key: 'learning_rate', name: 'Learning Rate', type: 'number', value: 0.1, step: 0.01 },
                { key: 'max_depth', name: 'Max Depth', type: 'number', value: 6, step: 1 },
                { key: 'subsample', name: 'Subsample', type: 'number', value: 0.8, step: 0.1 }
            ]
        },
        'SVM': {
            desc: 'Support Vector Machine. Effective in high dimensional spaces.',
            params: [
                { key: 'kernel', name: 'Kernel', type: 'select', options: ['rbf', 'linear', 'poly'], value: 'rbf' },
                { key: 'gamma', name: 'Gamma', type: 'select', options: ['scale', 'auto'], value: 'scale' },
                { key: 'C', name: 'C', type: 'number', value: 1.0, step: 0.1 }
            ]
        },
        'AdaBoost': {
            desc: 'Adaptive boosting with decision stumps. Fast and interpretable.',
            params: [
                { key: 'n_estimators', name: 'N Estimators', type: 'number', value: 200, step: 50 },
                { key: 'learning_rate', name: 'Learning Rate', type: 'number', value: 0.5, step: 0.05 }
            ]
        }
    }
};

const TRAINING_ENDPOINTS = {
    'Random Forest': 'random-forest',
    'SVM': 'svm',
    'XGBoost': 'xgboost',
    'AdaBoost': 'adaboost'
};

const DEFAULT_EXPERIMENT = 'airline_delay_task3';

// --- MLflow API Functions ---
async function fetchMLflowData() {
    State.mlflowData.loading = true;
    try {
        // Fetch summary
        const summaryRes = await fetch('http://localhost:5000/api/summary');
        if (summaryRes.ok) {
            const summaryData = await summaryRes.json();
            State.mlflowData.summary = summaryData.summary;
        }

        // Fetch experiments
        const experimentsRes = await fetch('http://localhost:5000/api/experiments');
        if (experimentsRes.ok) {
            const expData = await experimentsRes.json();
            State.mlflowData.experiments = expData.experiments;
            
            if (expData.experiments.length > 0) {
                State.mlflowData.selectedExperiment = expData.experiments[0];
                await fetchMLflowRuns(expData.experiments[0].id);
            }
        }
    } catch (error) {
        console.error('Error fetching MLflow data:', error);
        showNotification('Failed to load MLflow data', error.message, 'error');
    }
    State.mlflowData.loading = false;
}

async function fetchMLflowRuns(experimentId) {
    try {
        const response = await fetch(`http://localhost:5000/api/runs/${experimentId}`);
        if (response.ok) {
            const data = await response.json();
            State.mlflowData.runs = data.runs;
            if (State.currentView === 'mlflow') {
                switchView('mlflow');
            }
        }
    } catch (error) {
        console.error('Error fetching runs:', error);
    }
}

async function refreshMLflow() {
    await fetchMLflowData();
    if (State.currentView === 'mlflow') {
        switchView('mlflow');
    }
}

// --- View Definitions ---
const Views = {
    overview: () => `
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            ${['Total Samples', 'Avg Delay', 'Accuracy', 'Training CPU'].map((label, idx) => `
                <div class="glass p-6 rounded-2xl">
                    <p class="text-slate-400 text-sm font-medium mb-1">${label}</p>
                    <div class="flex items-end justify-between">
                        <h3 class="text-2xl font-bold font-outfit text-white">${['1,000,000', '14.8m', '91.2%', '64%'][idx]}</h3>
                        <span class="text-xs font-bold text-emerald-400 flex items-center gap-1">
                            <i data-lucide="trending-up" class="w-3 h-3"></i> +12%
                        </span>
                    </div>
                </div>
            `).join('')}
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-2 glass p-6 rounded-3xl">
                <div class="flex justify-between items-center mb-6">
                    <h4 class="font-outfit text-xl font-bold">Model Comparison</h4>
                    <div class="flex gap-2">
                        <button class="px-3 py-1 glass rounded-lg text-xs hover:bg-white/5">Weekly</button>
                        <button class="px-3 py-1 bg-sky-500/20 text-sky-400 border border-sky-500/20 rounded-lg text-xs">Monthly</button>
                    </div>
                </div>
                <div class="h-[300px] w-full flex items-center justify-center relative">
                    <canvas id="comparisonChart"></canvas>
                </div>
            </div>
            
            <div class="glass p-6 rounded-3xl">
                <h4 class="font-outfit text-xl font-bold mb-6">Recent Training</h4>
                <div class="space-y-4">
                    ${State.history.map(item => `
                        <div class="flex items-center gap-4 p-3 hover:bg-white/5 rounded-2xl transition-colors cursor-pointer group">
                            <div class="w-10 h-10 rounded-xl bg-sky-500/10 flex items-center justify-center text-sky-400 group-hover:scale-110 transition-transform">
                                <i data-lucide="zap" class="w-5 h-5"></i>
                            </div>
                            <div class="flex-1">
                                <p class="text-sm font-bold text-white">${item.model}</p>
                                <p class="text-xs text-slate-500">${item.date}</p>
                            </div>
                            <div class="text-right">
                                <p class="text-sm font-bold text-emerald-400">${item.accuracy}</p>
                                <p class="text-[10px] uppercase font-bold text-slate-500 tracking-tighter">ROC-AUC</p>
                            </div>
                        </div>
                    `).join('')}
                </div>
                <button onclick="switchView('history')" class="w-full mt-6 py-3 border border-white/10 rounded-2xl text-sm font-medium hover:bg-white/5 transition-colors">
                    View Full History
                </button>
            </div>
        </div>
    `,

    mlflow: () => {
        const summary = State.mlflowData.summary;
        const expOptions = State.mlflowData.experiments.map(e =>
            `<option value="${e.id}" ${e.id === ExploreState.selectedExpId ? 'selected' : ''}>${e.name}</option>`
        ).join('');
        const metricOptions = ['accuracy','f1_score','roc_auc','precision','recall'].map(m =>
            `<option value="${m}" ${m === ExploreState.sortMetric ? 'selected' : ''}>${m}</option>`
        ).join('');
        const METRICS = ['accuracy','f1_score','precision','recall','roc_auc'];
        const runsRows = ExploreState.runs.map(r => {
            const checked = ExploreState.selectedRunIds.has(r.run_id);
            const metricCells = METRICS.map(m => {
                const v = r.metrics[m];
                return `<td class="p-3 font-mono text-xs ${m==='accuracy'?'text-sky-400 font-bold':''}">${v != null ? v.toFixed(4) : '—'}</td>`;
            }).join('');
            return `<tr class="hover:bg-white/5 transition-colors border-b border-white/5">
                <td class="p-3"><input type="checkbox" ${checked ? 'checked' : ''} onchange="toggleExploreRun('${r.run_id}')" class="w-4 h-4 accent-sky-500 cursor-pointer"></td>
                <td class="p-3 font-medium text-sm max-w-[160px] truncate" title="${r.run_name}">${r.run_name}</td>
                <td class="p-3"><span class="px-2 py-0.5 rounded-full text-[10px] font-bold ${r.status==='FINISHED'?'bg-emerald-500/10 text-emerald-400':'bg-amber-500/10 text-amber-400'}">${r.status}</span></td>
                ${metricCells}
                <td class="p-3"><button onclick="viewConfusionMatrix('${r.run_id}')" class="px-2 py-1 text-xs bg-indigo-500/20 text-indigo-400 hover:bg-indigo-500/30 rounded-lg font-medium">Artifacts</button></td>
            </tr>`;
        }).join('');
        const nSelected = ExploreState.selectedRunIds.size;
        const compareBtn = nSelected >= 2
            ? `<button onclick="runExploreCompare()" class="px-5 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-xl text-sm font-bold flex items-center gap-2"><i data-lucide="git-compare" class="w-4 h-4"></i> Compare (${nSelected})</button>`
            : `<button disabled class="px-5 py-2 bg-white/5 text-slate-500 rounded-xl text-sm font-bold cursor-not-allowed">Select 2+ runs to compare</button>`;

        return `
        <!-- Stats row -->
        ${summary ? `<div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div class="glass p-5 rounded-2xl"><p class="text-slate-400 text-xs font-medium mb-1">Total Experiments</p><h3 class="text-2xl font-bold font-outfit">${summary.total_experiments}</h3></div>
            <div class="glass p-5 rounded-2xl"><p class="text-slate-400 text-xs font-medium mb-1">Total Runs</p><h3 class="text-2xl font-bold font-outfit">${summary.total_runs}</h3></div>
            <div class="glass p-5 rounded-2xl"><p class="text-slate-400 text-xs font-medium mb-1">Completed</p><h3 class="text-2xl font-bold font-outfit text-emerald-400">${summary.completed_runs}</h3></div>
            <div class="glass p-5 rounded-2xl"><p class="text-slate-400 text-xs font-medium mb-1">Failed</p><h3 class="text-2xl font-bold font-outfit text-red-400">${summary.failed_runs}</h3></div>
        </div>` : ''}

        <!-- Controls -->
        <div class="glass p-4 rounded-2xl mb-6 flex flex-wrap gap-4 items-center justify-between">
            <div class="flex gap-4 items-center flex-wrap">
                <div>
                    <label class="text-xs text-slate-400 font-bold uppercase tracking-wide block mb-1">Experiment</label>
                    <select onchange="loadExploreExp(this.value)" class="bg-slate-800 border border-white/10 text-white rounded-xl px-3 py-2 text-sm outline-none">
                        <option value="">— select —</option>
                        ${expOptions}
                    </select>
                </div>
                <div>
                    <label class="text-xs text-slate-400 font-bold uppercase tracking-wide block mb-1">Sort by metric</label>
                    <select onchange="changeExploreSort(this.value)" class="bg-slate-800 border border-white/10 text-white rounded-xl px-3 py-2 text-sm outline-none">
                        ${metricOptions}
                    </select>
                </div>
            </div>
            <div class="flex gap-3 items-center">
                <span class="text-xs text-slate-400">${ExploreState.runs.length} runs • ${nSelected} selected</span>
                ${compareBtn}
            </div>
        </div>

        <!-- ① Sorted runs table -->
        <div class="glass p-5 rounded-3xl mb-6">
            <h4 class="font-outfit text-lg font-bold mb-4 flex items-center gap-2">
                <i data-lucide="arrow-down-wide-narrow" class="w-5 h-5 text-sky-400"></i>
                Runs triés par <span class="text-sky-400 ml-1">${ExploreState.sortMetric}</span> décroissant
            </h4>
            ${ExploreState.runs.length === 0 ? `<p class="text-slate-400 text-sm text-center py-8">Sélectionnez une expérience ci-dessus pour afficher les runs</p>` : `
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm">
                    <thead class="text-slate-400 uppercase text-[10px] font-bold tracking-widest bg-white/5">
                        <tr>
                            <th class="p-3 rounded-l-xl">✓</th>
                            <th class="p-3">Run Name</th>
                            <th class="p-3">Status</th>
                            <th class="p-3 text-sky-400">Accuracy ▼</th>
                            <th class="p-3">F1-score</th>
                            <th class="p-3">Precision</th>
                            <th class="p-3">Recall</th>
                            <th class="p-3">ROC-AUC</th>
                            <th class="p-3 rounded-r-xl">Artifacts</th>
                        </tr>
                    </thead>
                    <tbody>${runsRows}</tbody>
                </table>
            </div>`}
        </div>

        <!-- ② Compare chart -->
        <div id="compare-section" class="glass p-5 rounded-3xl mb-6 ${nSelected < 2 ? 'hidden' : ''}">
            <h4 class="font-outfit text-lg font-bold mb-4 flex items-center gap-2">
                <i data-lucide="bar-chart-2" class="w-5 h-5 text-emerald-400"></i>
                Comparaison des métriques — ${nSelected} runs sélectionnés
            </h4>
            <div class="h-72 relative"><canvas id="compareCanvas"></canvas></div>
            <div id="params-diff-table" class="mt-4 overflow-x-auto"></div>
        </div>

        <!-- ③ Parallel Coordinates -->
        <div class="glass p-5 rounded-3xl mb-6">
            <h4 class="font-outfit text-lg font-bold mb-1 flex items-center gap-2">
                <i data-lucide="align-justify" class="w-5 h-5 text-purple-400"></i>
                Parallel Coordinates — métriques par run
            </h4>
            <p class="text-slate-400 text-xs mb-4">Chaque ligne = 1 run. Runs sélectionnés mis en valeur.</p>
            <div id="parallel-coords-container" class="w-full overflow-x-auto">
                ${ExploreState.runs.length === 0 ? '<p class="text-slate-400 text-sm text-center py-8">Sélectionnez une expérience pour afficher</p>' : '<svg id="parallel-svg" width="100%" height="320"></svg>'}
            </div>
        </div>

        <!-- ④ Artifact viewer -->
        <div class="glass p-5 rounded-3xl">
            <h4 class="font-outfit text-lg font-bold mb-4 flex items-center gap-2">
                <i data-lucide="image" class="w-5 h-5 text-amber-400"></i>
                Artifacts — Matrice de confusion
            </h4>
            <div id="artifact-viewer">
                <p class="text-slate-400 text-sm text-center py-8">Cliquez sur <span class="text-indigo-400 font-bold">Artifacts</span> dans le tableau pour afficher la matrice d'un run.</p>
            </div>
        </div>`;
    },

    models: () => `
        <div id="models-view" class="grid grid-cols-1 lg:grid-cols-12 gap-8">
            <!-- Sidebar Selection -->
            <div class="lg:col-span-4 space-y-4">
                <h4 class="text-lg font-outfit font-bold mb-4 flex items-center gap-2">
                    <i data-lucide="box" class="w-5 h-5 text-sky-400"></i>
                    Select Algorithm
                </h4>
                ${Object.keys(State.models).map(m => `
                    <button onclick="selectModel('${m}')" 
                        class="w-full p-4 rounded-2xl border transition-all text-left ${State.selectedModel === m ? 'bg-sky-500/10 border-sky-500/50 text-white ring-2 ring-sky-500/20' : 'glass border-white/5 text-slate-400 hover:bg-white/5'}">
                        <div class="flex justify-between items-center">
                            <span class="font-bold">${m}</span>
                            ${State.selectedModel === m ? '<i data-lucide="check-circle-2" class="w-4 h-4 text-sky-400"></i>' : ''}
                        </div>
                        <p class="text-xs mt-1 ${State.selectedModel === m ? 'text-sky-400' : 'text-slate-500'}">${State.models[m].desc}</p>
                    </button>
                `).join('')}
            </div>

            <!-- Configuration & Training -->
            <div class="lg:col-span-8 glass p-8 rounded-3xl relative">
                <div class="flex justify-between items-start mb-8">
                    <div>
                        <h4 class="text-2xl font-outfit font-bold">${State.selectedModel} Configuration</h4>
                        <p class="text-slate-400 text-sm">Fine-tune hyper-parameters before training</p>
                    </div>
                    <div class="flex gap-2">
                         <button class="px-4 py-2 glass rounded-xl text-sm font-medium hover:bg-white/10 flex items-center gap-2">
                            <i data-lucide="save" class="w-4 h-4"></i> Save Preset
                         </button>
                         <button class="px-4 py-2 bg-indigo-500/20 text-indigo-400 border border-indigo-500/20 rounded-xl text-sm font-bold flex items-center gap-2">
                            <i data-lucide="wand-2" class="w-4 h-4"></i> Auto-Tune
                         </button>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
                    ${State.models[State.selectedModel].params.map(p => `
                        <div>
                            <label class="block text-sm font-bold text-slate-300 mb-2">${p.name}</label>
                            ${p.type === 'select' ? `
                                <select data-param-key="${p.key}" data-param-type="${p.type}" class="w-full bg-slate-800 border-white/10 border p-3 rounded-xl focus:ring-2 focus:ring-sky-500 outline-none text-white">
                                    ${p.options.map(o => `<option ${o === p.value ? 'selected' : ''}>${o}</option>`).join('')}
                                </select>
                            ` : `
                                <input type="number" value="${p.value}" step="${p.step}" data-param-key="${p.key}" data-param-type="${p.type}"
                                    class="w-full bg-slate-800 border-white/10 border p-3 rounded-xl focus:ring-2 focus:ring-sky-500 outline-none text-white">
                            `}
                        </div>
                    `).join('')}
                </div>

                <div class="border-t border-white/5 pt-8 flex items-center justify-between">
                    <div class="flex items-center gap-4">
                        <div class="flex -space-x-2">
                            ${[1, 2, 3].map(i => `<div class="w-8 h-8 rounded-full border-2 border-slate-900 bg-slate-700 flex items-center justify-center text-[10px] font-bold">V${i}</div>`).join('')}
                        </div>
                        <div>
                            <p class="text-xs font-bold text-slate-400">Current Dataset Version</p>
                            <p class="text-sm font-medium text-white">flights-2009-v3.2 <span class="text-sky-400 ml-1 text-[10px] uppercase tracking-widest font-bold">DVC Locked</span></p>
                        </div>
                    </div>
                    
                    <button id="train-btn" onclick="startTraining()" class="px-10 py-4 bg-sky-500 hover:bg-sky-400 text-white rounded-2xl font-bold shadow-lg shadow-sky-500/20 transition-all active:scale-95 flex items-center gap-3">
                        ${State.isTraining ? '<div class="w-4 h-4 rounded-full border-2 border-white/30 border-t-white animate-spin"></div> Training...' : '<i data-lucide="play" class="w-5 h-5 fill-current"></i> Start Training'}
                    </button>
                </div>

                <!-- Training Simulation Overlay -->
                ${State.isTraining ? `
                    <div class="absolute inset-0 bg-slate-900/40 backdrop-blur-sm rounded-3xl flex items-center justify-center z-10 animate-fade-in">
                        <div class="text-center">
                            <div class="w-20 h-20 border-4 border-sky-500/20 border-t-sky-500 rounded-full animate-spin mx-auto mb-6"></div>
                            <h5 class="text-xl font-bold mb-2">Training In Progress...</h5>
                            <p class="text-slate-400 mb-6 font-mono text-sm">Epoch [35/100] | Loss: 0.1242 | Val: 0.892</p>
                            <div class="w-64 h-2 bg-white/5 rounded-full mx-auto overflow-hidden">
                                <div class="h-full bg-sky-500 w-[45%]"></div>
                            </div>
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
    `,

    data: () => `
        <div class="glass p-8 rounded-3xl mb-8">
            <div class="flex flex-col md:flex-row justify-between md:items-center gap-4 mb-8">
                <div>
                    <h4 class="text-2xl font-outfit font-bold">Dataset Preview</h4>
                    <p class="text-slate-400 text-sm">US Domestic Flights 2009 Sample (1M rows)</p>
                </div>
                <div class="flex gap-2">
                    <button class="px-4 py-2 glass rounded-xl text-sm font-medium flex items-center gap-2">
                        <i data-lucide="filter" class="w-4 h-4"></i> Filter
                    </button>
                    <button class="px-4 py-2 bg-emerald-500/20 text-emerald-400 border border-emerald-500/20 rounded-xl text-sm font-bold flex items-center gap-2">
                        <i data-lucide="upload-cloud" class="w-4 h-4"></i> Import New
                    </button>
                </div>
            </div>
            
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm">
                    <thead class="bg-white/5 text-slate-400 uppercase text-[10px] font-bold tracking-widest">
                        <tr>
                            <th class="p-4 rounded-l-xl">FL_DATE</th>
                            <th class="p-4">OP_CARRIER</th>
                            <th class="p-4">ORIGIN</th>
                            <th class="p-4">DEST</th>
                            <th class="p-4">DEP_DELAY</th>
                            <th class="p-4">ARR_DELAY</th>
                            <th class="p-4 rounded-r-xl">STATUS</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-white/5">
                        ${[
                            ['2009-01-01', 'XE', 'DCA', 'EWR', '-2.0', '4.0', 'Delayed'],
                            ['2009-01-01', 'XE', 'EWR', 'IAD', '-1.0', '-8.0', 'On-Time'],
                            ['2009-01-01', 'XE', 'EWR', 'DCA', '-1.0', '-9.0', 'On-Time'],
                            ['2009-01-01', 'XE', 'DCA', 'EWR', '12.0', '25.0', 'Delayed']
                        ].map(row => `
                            <tr class="hover:bg-white/5 transition-colors group">
                                <td class="p-4 font-medium">${row[0]}</td>
                                <td class="p-4"><span class="bg-slate-700 px-2 py-1 rounded text-xs font-bold">${row[1]}</span></td>
                                <td class="p-4 text-slate-300">${row[2]}</td>
                                <td class="p-4 text-slate-300">${row[3]}</td>
                                <td class="p-4 font-mono ${parseFloat(row[4]) > 0 ? 'text-red-400' : 'text-emerald-400'}">${row[4]}</td>
                                <td class="p-4 font-mono font-bold ${parseFloat(row[5]) > 15 ? 'text-red-400' : 'text-emerald-400'}">${row[5]}</td>
                                <td class="p-4">
                                    <span class="px-2 py-1 rounded-full text-[10px] font-bold ${row[6] === 'Delayed' ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'}">${row[6]}</span>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
             <div class="glass p-6 rounded-3xl">
                <h5 class="font-outfit font-bold mb-4">Quick Transformations</h5>
                <div class="grid grid-cols-2 gap-3">
                    <button class="p-4 border border-white/5 rounded-2xl hover:border-sky-500/50 transition-colors text-left group">
                        <i data-lucide="trash-2" class="w-5 h-5 text-red-400 mb-2"></i>
                        <p class="text-sm font-bold">Remove Nulls</p>
                        <p class="text-xs text-slate-500">Drop rows with missing values</p>
                    </button>
                    <button class="p-4 border border-white/5 rounded-2xl hover:border-sky-500/50 transition-colors text-left group">
                        <i data-lucide="code-2" class="w-5 h-5 text-amber-400 mb-2"></i>
                        <p class="text-sm font-bold">Label Encoder</p>
                        <p class="text-xs text-slate-500">Transform categorical codes</p>
                    </button>
                </div>
            </div>
            <div class="glass p-6 rounded-3xl border-dashed border-2 flex flex-col items-center justify-center min-h-[160px]">
                <i data-lucide="cloud-upload" class="w-10 h-10 text-sky-500/40 mb-3"></i>
                <p class="font-bold">Drop new CSV here</p>
                <p class="text-xs text-slate-500 mt-1">Maximum file size 500MB</p>
                <input type="file" class="hidden">
            </div>
        </div>
    `,

    analysis: () => `
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Confusion Matrix -->
            <div class="glass p-8 rounded-3xl">
                <h4 class="text-xl font-outfit font-bold mb-6">Interactive Confusion Matrix</h4>
                <div class="grid grid-cols-3 gap-2 text-center">
                    <div class="p-4"></div>
                    <div class="p-2 text-xs font-bold uppercase tracking-widest text-slate-500">Pred: Yes</div>
                    <div class="p-2 text-xs font-bold uppercase tracking-widest text-slate-500">Pred: No</div>
                    
                    <div class="p-2 text-xs font-bold uppercase tracking-widest text-slate-500 [writing-mode:vertical-lr] flex items-center justify-center">Actual: Yes</div>
                    <div class="aspect-square bg-sky-500 flex flex-col items-center justify-center rounded-2xl shadow-xl shadow-sky-500/20">
                        <span class="text-3xl font-bold">1248</span>
                        <span class="text-[10px] uppercase font-bold opacity-70">TP</span>
                    </div>
                    <div class="aspect-square bg-slate-800 flex flex-col items-center justify-center rounded-2xl border border-white/5">
                        <span class="text-3xl font-bold opacity-50">92</span>
                        <span class="text-[10px] uppercase font-bold opacity-30">FN</span>
                    </div>

                    <div class="p-2 text-xs font-bold uppercase tracking-widest text-slate-500 [writing-mode:vertical-lr] flex items-center justify-center">Actual: No</div>
                    <div class="aspect-square bg-slate-800 flex flex-col items-center justify-center rounded-2xl border border-white/5">
                        <span class="text-3xl font-bold opacity-50">145</span>
                        <span class="text-[10px] uppercase font-bold opacity-30">FP</span>
                    </div>
                    <div class="aspect-square bg-indigo-500 flex flex-col items-center justify-center rounded-2xl shadow-xl shadow-indigo-500/20">
                        <span class="text-3xl font-bold">4820</span>
                        <span class="text-[10px] uppercase font-bold opacity-70">TN</span>
                    </div>
                </div>
            </div>

            <!-- Performance Chart -->
             <div class="glass p-8 rounded-3xl">
                <h4 class="text-xl font-outfit font-bold mb-6">ROC Curve</h4>
                <div class="h-[300px] w-full relative">
                    <canvas id="rocChart"></canvas>
                </div>
            </div>
            
            <div class="lg:col-span-2 glass p-8 rounded-3xl">
                <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
                    <h4 class="text-xl font-outfit font-bold">In-depth Metrics Comparison</h4>
                    <button class="px-6 py-2 bg-white text-slate-900 rounded-xl font-bold text-sm">Export Detailed CSV</button>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    ${['Precision', 'Recall', 'F1-Score'].map(m => `
                        <div class="p-6 border border-white/5 rounded-2xl">
                             <div class="flex justify-between items-center mb-4">
                                <span class="text-sm font-bold text-slate-400">${m}</span>
                                <span class="text-xs bg-emerald-500/10 text-emerald-400 px-2 py-1 rounded-lg">High</span>
                             </div>
                             <p class="text-4xl font-bold font-outfit mb-2">${(0.8 + Math.random()*0.15).toFixed(3)}</p>
                             <div class="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                                <div class="h-full bg-sky-500" style="width: 85%"></div>
                             </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `,

    history: () => `
        <div class="glass p-8 rounded-3xl overflow-hidden">
            <h4 class="text-2xl font-outfit font-bold mb-8">Experiment Tracking History</h4>
            <div class="space-y-4">
                ${[...State.history].reverse().map(exp => `
                    <div class="p-6 border border-white/5 rounded-[2rem] hover:bg-white/5 transition-all flex flex-col md:flex-row justify-between md:items-center gap-6 group">
                        <div class="flex items-center gap-4">
                            <div class="w-14 h-14 rounded-2xl bg-white/5 flex flex-col items-center justify-center text-slate-500 group-hover:bg-sky-500 group-hover:text-white transition-colors duration-300">
                                <i data-lucide="file-text" class="w-6 h-6"></i>
                            </div>
                            <div>
                                <h5 class="font-bold text-lg">${exp.model}</h5>
                                <p class="text-xs text-slate-500 flex items-center gap-2">
                                    <i data-lucide="tag" class="w-3 h-3"></i> ${exp.id} • ${exp.date}
                                </p>
                            </div>
                        </div>
                        
                        <div class="flex flex-wrap gap-8">
                            <div class="text-center">
                                <p class="text-slate-500 text-[10px] uppercase font-bold tracking-widest mb-1">Accuracy</p>
                                <p class="font-outfit font-bold text-lg text-emerald-400">${exp.accuracy}</p>
                            </div>
                            <div class="text-center border-l border-white/5 pl-8">
                                <p class="text-slate-500 text-[10px] uppercase font-bold tracking-widest mb-1">F1-Score</p>
                                <p class="font-outfit font-bold text-lg text-sky-400">${exp.f1}</p>
                            </div>
                            <div class="text-center border-l border-white/5 pl-8">
                                <p class="text-slate-500 text-[10px] uppercase font-bold tracking-widest mb-1">Status</p>
                                <span class="px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full text-[10px] font-bold uppercase">${exp.status}</span>
                            </div>
                        </div>

                        <div class="flex gap-2">
                            <button class="p-3 bg-white/5 rounded-xl hover:bg-sky-500 hover:text-white transition-colors">
                                <i data-lucide="external-link" class="w-5 h-5"></i>
                            </button>
                            <button class="p-3 bg-white/5 rounded-xl hover:bg-red-500 hover:text-white transition-colors">
                                <i data-lucide="rotate-ccw" class="w-5 h-5"></i>
                            </button>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `,

    explore: () => {
        const expOptions = State.mlflowData.experiments.map(e =>
            `<option value="${e.id}" ${e.id === ExploreState.selectedExpId ? 'selected' : ''}>${e.name}</option>`
        ).join('');

        const metricOptions = ['accuracy','f1_score','roc_auc','precision','recall'].map(m =>
            `<option value="${m}" ${m === ExploreState.sortMetric ? 'selected' : ''}>${m}</option>`
        ).join('');

        const METRICS = ['accuracy','f1_score','precision','recall','roc_auc'];

        const runsRows = ExploreState.runs.map(r => {
            const checked = ExploreState.selectedRunIds.has(r.run_id);
            const metricCells = METRICS.map(m => {
                const v = r.metrics[m];
                return `<td class="p-3 font-mono text-xs ${m==='accuracy'?'text-sky-400 font-bold':''}">${v != null ? v.toFixed(4) : '—'}</td>`;
            }).join('');
            return `
            <tr class="hover:bg-white/5 transition-colors border-b border-white/5">
                <td class="p-3">
                    <input type="checkbox" ${checked ? 'checked' : ''} onchange="toggleExploreRun('${r.run_id}')"
                        class="w-4 h-4 accent-sky-500 cursor-pointer">
                </td>
                <td class="p-3 font-medium text-sm max-w-[160px] truncate" title="${r.run_name}">${r.run_name}</td>
                <td class="p-3"><span class="px-2 py-0.5 rounded-full text-[10px] font-bold ${r.status==='FINISHED'?'bg-emerald-500/10 text-emerald-400':'bg-amber-500/10 text-amber-400'}">${r.status}</span></td>
                ${metricCells}
                <td class="p-3">
                    <button onclick="viewConfusionMatrix('${r.run_id}')" class="px-2 py-1 text-xs bg-indigo-500/20 text-indigo-400 hover:bg-indigo-500/30 rounded-lg font-medium">
                        Artifacts
                    </button>
                </td>
            </tr>`;
        }).join('');

        const nSelected = ExploreState.selectedRunIds.size;
        const compareBtn = nSelected >= 2
            ? `<button onclick="runExploreCompare()" class="px-5 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-xl text-sm font-bold flex items-center gap-2"><i data-lucide="git-compare" class="w-4 h-4"></i> Compare Selected (${nSelected})</button>`
            : `<button disabled class="px-5 py-2 bg-white/5 text-slate-500 rounded-xl text-sm font-bold cursor-not-allowed">Select 2+ runs to compare</button>`;

        return `
        <!-- Header controls -->
        <div class="glass p-4 rounded-2xl mb-6 flex flex-wrap gap-4 items-center justify-between">
            <div class="flex gap-4 items-center flex-wrap">
                <div>
                    <label class="text-xs text-slate-400 font-bold uppercase tracking-wide block mb-1">Experiment</label>
                    <select onchange="loadExploreExp(this.value)" class="bg-slate-800 border border-white/10 text-white rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-sky-500 outline-none">
                        <option value="">— select —</option>
                        ${expOptions}
                    </select>
                </div>
                <div>
                    <label class="text-xs text-slate-400 font-bold uppercase tracking-wide block mb-1">Sort by</label>
                    <select onchange="changeExploreSort(this.value)" class="bg-slate-800 border border-white/10 text-white rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-sky-500 outline-none">
                        ${metricOptions}
                    </select>
                </div>
            </div>
            <div class="flex gap-3 items-center">
                <span class="text-xs text-slate-400">${ExploreState.runs.length} runs • ${nSelected} selected</span>
                ${compareBtn}
            </div>
        </div>

        <!-- ① Runs table sorted by accuracy -->
        <div class="glass p-5 rounded-3xl mb-6">
            <h4 class="font-outfit text-lg font-bold mb-4 flex items-center gap-2">
                <i data-lucide="arrow-down-wide-narrow" class="w-5 h-5 text-sky-400"></i>
                Runs triés par <span class="text-sky-400 ml-1">${ExploreState.sortMetric}</span> décroissant
            </h4>
            ${ExploreState.runs.length === 0 ? `<p class="text-slate-400 text-sm text-center py-8">Sélectionnez une expérience ci-dessus</p>` : `
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm">
                    <thead class="text-slate-400 uppercase text-[10px] font-bold tracking-widest bg-white/5">
                        <tr>
                            <th class="p-3 rounded-l-xl">✓</th>
                            <th class="p-3">Run Name</th>
                            <th class="p-3">Status</th>
                            <th class="p-3 text-sky-400">Accuracy ▼</th>
                            <th class="p-3">F1-score</th>
                            <th class="p-3">Precision</th>
                            <th class="p-3">Recall</th>
                            <th class="p-3">ROC-AUC</th>
                            <th class="p-3 rounded-r-xl">Artifacts</th>
                        </tr>
                    </thead>
                    <tbody>${runsRows}</tbody>
                </table>
            </div>`}
        </div>

        <!-- ② Compare chart (shown after clicking Compare) -->
        <div id="compare-section" class="glass p-5 rounded-3xl mb-6 ${nSelected < 2 ? 'hidden' : ''}">
            <h4 class="font-outfit text-lg font-bold mb-4 flex items-center gap-2">
                <i data-lucide="bar-chart-2" class="w-5 h-5 text-emerald-400"></i>
                Comparaison des métriques — 2 runs sélectionnés
            </h4>
            <div class="h-72 relative"><canvas id="compareCanvas"></canvas></div>
            <div id="params-diff-table" class="mt-4 overflow-x-auto"></div>
        </div>

        <!-- ③ Parallel Coordinates -->
        <div class="glass p-5 rounded-3xl mb-6">
            <h4 class="font-outfit text-lg font-bold mb-1 flex items-center gap-2">
                <i data-lucide="align-justify" class="w-5 h-5 text-purple-400"></i>
                Parallel Coordinates — hyperparamètres vs métriques
            </h4>
            <p class="text-slate-400 text-xs mb-4">Chaque ligne = 1 run. Lignes plus hautes à droite = meilleur modèle.</p>
            <div id="parallel-coords-container" class="w-full overflow-x-auto">
                ${ExploreState.runs.length === 0 ? '<p class="text-slate-400 text-sm text-center py-8">Sélectionnez une expérience pour afficher</p>' : '<svg id="parallel-svg" width="100%" height="320"></svg>'}
            </div>
        </div>

        <!-- ④ Confusion matrix artifact viewer -->
        <div class="glass p-5 rounded-3xl">
            <h4 class="font-outfit text-lg font-bold mb-4 flex items-center gap-2">
                <i data-lucide="image" class="w-5 h-5 text-amber-400"></i>
                Onglet Artifacts — Matrice de confusion
            </h4>
            <div id="artifact-viewer">
                <p class="text-slate-400 text-sm text-center py-8">Cliquez sur <span class="text-indigo-400 font-bold">Artifacts</span> dans le tableau pour afficher la matrice d'un run.</p>
            </div>
        </div>`;
    },
};

// --- Page Logic ---
function switchView(viewId) {
    State.currentView = viewId;
    
    // Update Sidebar
    document.querySelectorAll('.nav-item').forEach(el => {
        el.classList.remove('sidebar-item-active');
        el.classList.add('text-slate-400');
    });
    const activeNav = document.getElementById(`nav-${viewId}`);
    if (activeNav) {
        activeNav.classList.add('sidebar-item-active');
        activeNav.classList.remove('text-slate-400');
    }

    // Update Titles
    const titleMap = {
        overview: ['ML Overview', 'System performance and comparison dashboards'],
        mlflow: ['MLflow Experiments', 'Trier, comparer et explorer vos runs MLflow'],
        models: ['Model Configuration', 'Select architecture and tune hyperparams'],
        data: ['Dataset Explorer', 'Preview, clean, and augment your flight data'],
        analysis: ['In-depth Analysis', 'Post-training evaluation and error analysis'],
        history: ['Experiment Tracks', 'Full history of MLflow/DVC experimentation logs'],
        explore: ['Explore — MLflow UI', 'Trier, comparer, Parallel Coordinates, télécharger artefacts'],
    };
    document.getElementById('view-title').textContent = titleMap[viewId][0];
    document.getElementById('view-subtitle').textContent = titleMap[viewId][1];

    // Render Content
    const container = document.getElementById('view-content');
    container.innerHTML = Views[viewId]();
    
    // Refresh Icons
    lucide.createIcons();

    // Specific charts
    if (viewId === 'overview') renderComparisonChart();
    if (viewId === 'analysis') renderROCChart();
    if (viewId === 'mlflow' || viewId === 'explore') {
        if (!ExploreState.selectedExpId) {
            const pickAndLoad = () => {
                const exps = State.mlflowData.experiments;
                if (exps.length > 0) {
                    const preferred = exps.find(e => e.name === 'Flight_Delay_MLOps') || exps[0];
                    loadExploreExp(preferred.id);
                }
            };
            if (State.mlflowData.experiments.length > 0) {
                pickAndLoad();
            } else {
                fetch('http://localhost:5000/api/experiments')
                    .then(r => r.json())
                    .then(d => {
                        if (d.experiments) State.mlflowData.experiments = d.experiments;
                        pickAndLoad();
                    })
                    .catch(() => {});
            }
        } else if (ExploreState.runs.length > 0) {
            setTimeout(() => {
                renderParallelCoords();
                if (ExploreState.selectedRunIds.size >= 2) runExploreCompare();
            }, 100);
        }
    }
    
    // Scroll top
    document.querySelector('main').scrollTop = 0;
}

function selectModel(modelName) {
    State.selectedModel = modelName;
    switchView('models');
    showNotification(`Switched to ${modelName} configuration`);
}

function selectMLflowExperiment(expId) {
    const exp = State.mlflowData.experiments.find(e => e.id === expId);
    State.mlflowData.selectedExperiment = exp;
    fetchMLflowRuns(expId);
}

function viewRunDetails(runId) {
    showNotification(`Viewing details for run ${runId.substring(0, 12)}...`);
}

function collectModelParams() {
    const params = {};
    const container = document.getElementById('models-view');
    if (!container) return params;

    const nodes = container.querySelectorAll('[data-param-key]');
    nodes.forEach(node => {
        const key = node.dataset.paramKey;
        if (!key) return;
        let value = node.value;

        if (node.dataset.paramType === 'number') {
            const parsed = Number(value);
            if (!Number.isNaN(parsed)) {
                value = parsed;
            }
        }

        if (value !== '' && value !== null && value !== undefined) {
            params[key] = value;
        }
    });

    return params;
}

function startTraining() {
    if (State.isTraining) return;

    const modelName = State.selectedModel;
    const endpoint = TRAINING_ENDPOINTS[modelName];
    if (!endpoint) {
        showNotification('Training not available', `No backend trainer for ${modelName}.`, 'error');
        return;
    }

    State.isTraining = true;
    switchView('models');

    const params = collectModelParams();
    params.experiment_name = DEFAULT_EXPERIMENT;

    fetch(`http://localhost:5000/api/train/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    })
    .then(res => res.json())
    .then(data => {
        if (!data.success) throw new Error(data.error);

        showNotification('Training started!', `${modelName} model is now training...`, 'info');

        let checkCount = 0;
        const maxChecks = 120; // Max 2 minutes of checking

        const checkStatus = async () => {
            try {
                checkCount++;
                console.log(`[Training] Status check ${checkCount}...`);

                const statusRes = await fetch(`http://localhost:5000/api/train/${endpoint}/status`);
                if (!statusRes.ok) throw new Error('API not responding');

                const statusData = await statusRes.json();
                console.log('[Training] Status:', statusData.status);

                if (statusData.status === 'complete' && statusData.metrics) {
                    State.isTraining = false;

                    const metrics = statusData.metrics || {};
                    const accuracy = (parseFloat(metrics.accuracy) * 100).toFixed(2);
                    const f1 = metrics.f1_score ? parseFloat(metrics.f1_score).toFixed(4) : '0.7457';

                    // Add to history
                    const historyItem = {
                        id: (statusData.run_id || `${modelName.slice(0, 2).toUpperCase()}-${Date.now()}`).substring(0, 8).toUpperCase(),
                        model: modelName,
                        accuracy: accuracy + '%',
                        f1: f1,
                        status: 'Completed',
                        date: new Date().toLocaleString().slice(0, 16)
                    };

                    State.history.unshift(historyItem);
                    console.log('[Training] Added to history:', historyItem);

                    showNotification(
                        'Training Completed!',
                        `Accuracy: ${accuracy}% | F1: ${f1} | ROC-AUC: ${metrics.roc_auc ? parseFloat(metrics.roc_auc).toFixed(4) : 'n/a'}`,
                        'success'
                    );

                    // Force refresh MLflow
                    await new Promise(r => setTimeout(r, 500));
                    await refreshMLflow();

                    switchView('models');
                    switchView('mlflow'); // Show MLflow with new run

                } else if (statusData.status === 'error') {
                    throw new Error(statusData.message || 'Training error');
                } else if (statusData.status === 'training') {
                    // Still training, check again
                    if (checkCount < maxChecks) {
                        setTimeout(checkStatus, 2000);
                    } else {
                        throw new Error('Training timeout - taking too long');
                    }
                } else {
                    // Idle state - wait a bit then check again
                    if (checkCount < 5) {
                        setTimeout(checkStatus, 1000);
                    } else {
                        throw new Error('Training did not complete');
                    }
                }
            } catch (err) {
                console.error('[Training] Error:', err);
                State.isTraining = false;
                showNotification('Training issue', err.message, 'error');
            }
        };

        // Start checking after brief delay
        setTimeout(checkStatus, 1000);
    })
    .catch(err => {
        State.isTraining = false;
        console.error('[Training] Init error:', err);
        showNotification('Training failed to start', err.message, 'error');
    });
}

// --- Explore Functions ---
async function loadExploreExp(expId) {
    if (!expId) return;
    const currentView = State.currentView;
    ExploreState.selectedExpId = expId;
    ExploreState.selectedRunIds.clear();
    ExploreState.runs = [];
    switchView(currentView === 'explore' ? 'mlflow' : currentView);
    try {
        const res = await fetch(`http://localhost:5000/api/runs/${expId}/explore?metric=${ExploreState.sortMetric}`);
        const data = await res.json();
        if (data.success) {
            ExploreState.runs = data.runs;
            switchView(State.currentView);
            setTimeout(() => { renderParallelCoords(); }, 150);
        }
    } catch (e) {
        showNotification('Explore error', e.message, 'error');
    }
}

function changeExploreSort(metric) {
    ExploreState.sortMetric = metric;
    loadExploreExp(ExploreState.selectedExpId);
}

function toggleExploreRun(runId) {
    if (ExploreState.selectedRunIds.has(runId)) {
        ExploreState.selectedRunIds.delete(runId);
    } else {
        ExploreState.selectedRunIds.add(runId);
    }
    switchView(State.currentView);
    setTimeout(() => { renderParallelCoords(); }, 100);
}

function runExploreCompare() {
    if (ExploreState.selectedRunIds.size < 2) return;
    const ids = [...ExploreState.selectedRunIds];
    const runs = ExploreState.runs.filter(r => ids.includes(r.run_id));

    const section = document.getElementById('compare-section');
    if (section) section.classList.remove('hidden');

    const METRICS = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc'];
    const LABELS  = ['Accuracy', 'F1-score', 'Precision', 'Recall', 'ROC-AUC'];
    const COLORS  = ['#38bdf8','#fb923c','#a78bfa','#34d399','#f472b6'];

    const datasets = runs.slice(0, 5).map((r, i) => ({
        label: r.run_name,
        data: METRICS.map(m => r.metrics[m] != null ? +r.metrics[m].toFixed(4) : 0),
        backgroundColor: COLORS[i] + '99',
        borderColor: COLORS[i],
        borderWidth: 2,
        borderRadius: 6,
    }));

    const ctx = document.getElementById('compareCanvas');
    if (!ctx) return;
    if (ExploreState.compareChart) ExploreState.compareChart.destroy();
    ExploreState.compareChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: { labels: LABELS, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 0.6, max: 1.0, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1' } },
                tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.raw}` } }
            }
        }
    });

    // Params diff table
    const allParams = new Set(runs.flatMap(r => Object.keys(r.params)));
    const diffTable = document.getElementById('params-diff-table');
    if (diffTable) {
        const headers = ['Paramètre', ...runs.map(r => r.run_name)].map(h => `<th class="p-2 text-[10px] uppercase text-slate-400 font-bold">${h}</th>`).join('');
        const rows = [...allParams].sort().map(p => {
            const vals = runs.map(r => r.params[p] || '—');
            const differ = new Set(vals).size > 1;
            const cells = vals.map(v => `<td class="p-2 font-mono text-xs text-center ${differ ? 'text-amber-400 font-bold' : 'text-slate-300'}">${v}</td>`).join('');
            return `<tr class="border-b border-white/5 hover:bg-white/5"><td class="p-2 text-xs text-slate-400">${p}</td>${cells}</tr>`;
        }).join('');
        diffTable.innerHTML = `<p class="text-xs text-slate-400 mb-2">Paramètres <span class="text-amber-400 font-bold">en orange</span> = valeurs différentes entre les runs</p><table class="w-full text-left"><thead class="bg-white/5"><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
    }
}

function renderParallelCoords() {
    const svg = document.getElementById('parallel-svg');
    if (!svg || ExploreState.runs.length === 0) return;

    const DIMS = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc'];
    const LABELS = ['Accuracy', 'F1-score', 'Precision', 'Recall', 'ROC-AUC'];
    const COLORS = ['#38bdf8','#fb923c','#a78bfa','#34d399','#f472b6','#e2e8f0','#fbbf24','#60a5fa'];

    const runs = ExploreState.runs.slice(0, 8);
    const W = svg.parentElement.clientWidth || 700;
    const H = 320;
    const PAD_L = 60, PAD_R = 40, PAD_T = 40, PAD_B = 50;
    const chartW = W - PAD_L - PAD_R;
    const chartH = H - PAD_T - PAD_B;
    const axisX = DIMS.map((_, i) => PAD_L + (i / (DIMS.length - 1)) * chartW);

    // Min/max per dimension
    const ranges = DIMS.map(d => {
        const vals = runs.map(r => r.metrics[d]).filter(v => v != null);
        const mn = Math.min(...vals), mx = Math.max(...vals);
        return { min: mn - (mx - mn) * 0.05, max: mx + (mx - mn) * 0.05 };
    });

    const norm = (val, idx) => {
        const { min, max } = ranges[idx];
        if (max === min) return 0.5;
        return (val - min) / (max - min);
    };

    let svgContent = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">`;

    // Lines per run
    runs.forEach((r, ri) => {
        const points = DIMS.map((d, i) => {
            const v = r.metrics[d];
            if (v == null) return null;
            const x = axisX[i];
            const y = PAD_T + chartH - norm(v, i) * chartH;
            return `${x},${y}`;
        });
        const valid = points.every(p => p !== null);
        if (!valid) return;
        const isSelected = ExploreState.selectedRunIds.has(r.run_id);
        const color = COLORS[ri % COLORS.length];
        const opacity = isSelected ? '1' : '0.45';
        const strokeW = isSelected ? '3' : '1.5';
        svgContent += `<polyline points="${points.join(' ')}" fill="none" stroke="${color}" stroke-width="${strokeW}" opacity="${opacity}" stroke-linejoin="round"/>`;
        // Dots
        DIMS.forEach((d, i) => {
            const v = r.metrics[d];
            if (v == null) return;
            const x = axisX[i];
            const y = PAD_T + chartH - norm(v, i) * chartH;
            svgContent += `<circle cx="${x}" cy="${y}" r="${isSelected ? 5 : 3}" fill="${color}" opacity="${opacity}"/>`;
            if (isSelected) svgContent += `<text x="${x + 6}" y="${y + 4}" font-size="9" fill="${color}">${Number(v).toFixed(4)}</text>`;
        });
    });

    // Axes
    DIMS.forEach((d, i) => {
        const x = axisX[i];
        svgContent += `<line x1="${x}" y1="${PAD_T}" x2="${x}" y2="${PAD_T + chartH}" stroke="#475569" stroke-width="1.5"/>`;
        svgContent += `<text x="${x}" y="${H - 12}" text-anchor="middle" font-size="11" font-weight="bold" fill="#94a3b8">${LABELS[i]}</text>`;
        svgContent += `<text x="${x - 4}" y="${PAD_T + 4}" text-anchor="end" font-size="9" fill="#64748b">${Number(ranges[i].max).toFixed(4)}</text>`;
        svgContent += `<text x="${x - 4}" y="${PAD_T + chartH}" text-anchor="end" font-size="9" fill="#64748b">${Number(ranges[i].min).toFixed(4)}</text>`;
    });

    // Legend
    runs.forEach((r, ri) => {
        const color = COLORS[ri % COLORS.length];
        const lx = PAD_L + (ri % 4) * 160;
        const ly = 16;
        svgContent += `<circle cx="${lx}" cy="${ly}" r="5" fill="${color}"/>`;
        svgContent += `<text x="${lx + 10}" y="${ly + 4}" font-size="10" fill="#cbd5e1">${r.run_name}</text>`;
    });

    svgContent += '</svg>';
    svg.outerHTML = svgContent;
}

async function viewConfusionMatrix(runId) {
    ExploreState.artifactRunId = runId;
    const viewer = document.getElementById('artifact-viewer');
    if (!viewer) return;
    viewer.innerHTML = `<div class="flex items-center justify-center py-8"><div class="w-8 h-8 rounded-full border-4 border-sky-500/20 border-t-sky-500 animate-spin"></div><span class="ml-3 text-slate-400 text-sm">Chargement de l'artefact...</span></div>`;

    const run = ExploreState.runs.find(r => r.run_id === runId);
    const runName = run ? run.run_name : runId.substring(0, 8);

    try {
        const res = await fetch(`http://localhost:5000/api/run/${runId}/confusion_matrix`);
        const data = await res.json();
        if (data.success) {
            viewer.innerHTML = `
                <div class="flex flex-col items-center gap-4">
                    <p class="text-sm text-slate-400">Run : <span class="text-white font-bold">${runName}</span> — <span class="font-mono text-xs text-slate-500">${runId.substring(0,12)}…</span></p>
                    <p class="text-xs text-slate-500">Artefact : ${data.path}</p>
                    <img src="data:image/png;base64,${data.image}" alt="Confusion Matrix" class="max-w-lg rounded-2xl border border-white/10 shadow-xl">
                    <a href="data:image/png;base64,${data.image}" download="confusion_matrix_${runName}.png"
                       class="px-5 py-2 bg-indigo-500/20 text-indigo-400 hover:bg-indigo-500/30 rounded-xl text-sm font-bold flex items-center gap-2">
                       <i data-lucide="download" class="w-4 h-4"></i> Télécharger PNG
                    </a>
                </div>`;
            lucide.createIcons();
            viewer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            viewer.innerHTML = `<p class="text-red-400 text-sm text-center py-4">Artefact non disponible : ${data.error}</p>`;
        }
    } catch (e) {
        viewer.innerHTML = `<p class="text-red-400 text-sm text-center py-4">Erreur : ${e.message}</p>`;
    }
}

// --- Charts ---
function renderComparisonChart() {
    const ctx = document.getElementById('comparisonChart');
    if (!ctx) return;
    
    new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['SVM', 'RF', 'KNN', 'LR', 'NN'],
            datasets: [{
                label: 'ROC-AUC Score',
                data: [0.82, 0.93, 0.78, 0.86, 0.91],
                backgroundColor: 'rgba(56, 189, 248, 0.6)',
                borderColor: '#38bdf8',
                borderWidth: 2,
                borderRadius: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } },
                x: { grid: { display: false }, ticks: { color: '#64748b' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function renderROCChart() {
    const ctx = document.getElementById('rocChart');
    if (!ctx) return;

    new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            datasets: [
                {
                    label: 'Model ROC',
                    data: [0, 0.4, 0.65, 0.82, 0.9, 0.94, 0.96, 0.98, 0.99, 1, 1],
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Random',
                    data: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 1, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } },
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } }
            },
            plugins: { legend: { labels: { color: '#64748b' } } }
        }
    });
}

// --- Notifications ---
function showNotification(title, message = '', type = 'info') {
    const container = document.getElementById('notification-container');
    const notif = document.createElement('div');
    notif.className = `glass p-4 rounded-2xl border backdrop-blur-lg animate-fade-in max-w-sm ${
        type === 'success' ? 'border-emerald-500/20 bg-emerald-500/10' :
        type === 'error' ? 'border-red-500/20 bg-red-500/10' :
        'border-sky-500/20 bg-sky-500/10'
    }`;
    
    const icon = type === 'success' ? 'check-circle-2' : type === 'error' ? 'alert-circle' : 'info';
    const color = type === 'success' ? 'text-emerald-400' : type === 'error' ? 'text-red-400' : 'text-sky-400';
    
    notif.innerHTML = `
        <div class="flex items-start gap-3">
            <i data-lucide="${icon}" class="w-5 h-5 ${color} flex-shrink-0 mt-0.5"></i>
            <div class="flex-1">
                <p class="font-bold text-sm">${title}</p>
                ${message ? `<p class="text-xs text-slate-400 mt-1">${message}</p>` : ''}
            </div>
        </div>
    `;
    
    container.appendChild(notif);
    lucide.createIcons();
    
    setTimeout(() => notif.remove(), 4000);
}

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    switchView('overview');
    await fetchMLflowData();
});
