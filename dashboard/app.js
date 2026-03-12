// --- App State ---
const State = {
    currentView: 'overview',
    selectedModel: 'Random Forest',
    isTraining: false,
    history: [
        { id: 'EXP-001', model: 'Logistic Regression', accuracy: '84.2%', f1: '0.78', status: 'Completed', date: '2026-03-01 09:12' },
        { id: 'EXP-002', model: 'Random Forest', accuracy: '89.5%', f1: '0.86', status: 'Completed', date: '2026-03-01 10:45' },
    ],
    models: {
        'Logistic Regression': {
            desc: 'Linear model for binary classification. Efficient and interpretable.',
            params: [
                { name: 'Penalty', type: 'select', options: ['l2', 'l1', 'elasticnet'], value: 'l2' },
                { name: 'C (Inverse Regularization)', type: 'number', value: 1.0, step: 0.1 },
                { name: 'Solver', type: 'select', options: ['lbfgs', 'liblinear', 'saga'], value: 'lbfgs' }
            ]
        },
        'Random Forest': {
            desc: 'Ensemble of decision trees. Robust to noise and non-linear patterns.',
            params: [
                { name: 'N Estimators', type: 'number', value: 100, step: 10 },
                { name: 'Max Depth', type: 'number', value: 12, step: 1 },
                { name: 'Min Samples Split', type: 'number', value: 2, step: 1 }
            ]
        },
        'XGBoost': {
            desc: 'Gradient Boosted Decision Trees. High performance, used for structured data.',
            params: [
                { name: 'Learning Rate', type: 'number', value: 0.1, step: 0.01 },
                { name: 'Max Depth', type: 'number', value: 6, step: 1 },
                { name: 'Subsample', type: 'number', value: 0.8, step: 0.1 }
            ]
        },
        'SVM': {
            desc: 'Support Vector Machine. Effective in high dimensional spaces.',
            params: [
                { name: 'Kernel', type: 'select', options: ['rbf', 'linear', 'poly'], value: 'rbf' },
                { name: 'Gamma', type: 'select', options: ['scale', 'auto'], value: 'scale' },
                { name: 'C', type: 'number', value: 1.0, step: 0.1 }
            ]
        }
    }
};

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

    models: () => `
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
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
                                <select class="w-full bg-slate-800 border-white/10 border p-3 rounded-xl focus:ring-2 focus:ring-sky-500 outline-none text-white">
                                    ${p.options.map(o => `<option ${o === p.value ? 'selected' : ''}>${o}</option>`).join('')}
                                </select>
                            ` : `
                                <input type="number" value="${p.value}" step="${p.step}" 
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
    `
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
        models: ['Model Configuration', 'Select architecture and tune hyperparams'],
        data: ['Dataset Explorer', 'Preview, clean, and augment your flight data'],
        analysis: ['In-depth Analysis', 'Post-training evaluation and error analysis'],
        history: ['Experiment Tracks', 'Full history of MLflow/DVC experimentation logs']
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
    
    // Scroll top
    document.querySelector('main').scrollTop = 0;
}

function selectModel(modelName) {
    State.selectedModel = modelName;
    switchView('models');
    showNotification(`Switched to ${modelName} configuration`);
}

function startTraining() {
    if (State.isTraining) return;
    State.isTraining = true;
    switchView('models');
    
    setTimeout(() => {
        State.isTraining = false;
        const newExp = {
            id: `EXP-00${State.history.length + 1}`,
            model: State.selectedModel,
            accuracy: (90 + Math.random()*5).toFixed(1) + '%',
            f1: (0.85 + Math.random()*0.1).toFixed(2),
            status: 'Completed',
            date: new Date().toISOString().slice(0, 16).replace('T', ' ')
        };
        State.history.push(newExp);
        showNotification('Training Completed!', 'Model results have been saved to history.', 'success');
        switchView('models');
    }, 3000);
}

// --- Charts ---
function renderComparisonChart() {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    new Chart(ctx, {
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
    const ctx = document.getElementById('rocChart').getContext('2d');
    new Chart(ctx, {
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
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } },
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } }
            },
            plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8' } } }
        }
    });
}

// --- Notifications ---
function showNotification(title, message = '', type = 'info') {
    const container = document.getElementById('notification-container');
    const note = document.createElement('div');
    note.className = 'glass p-4 rounded-2xl w-80 shadow-2xl border-l-4 transform transition-all duration-300 translate-x-10 opacity-0 ' + (type === 'success' ? 'border-emerald-500' : 'border-sky-500');
    
    note.innerHTML = `
        <div class="flex gap-3">
            <div class="${type === 'success' ? 'text-emerald-500' : 'text-sky-500'}">
                <i data-lucide="${type === 'success' ? 'check-circle' : 'info'}" class="w-5 h-5"></i>
            </div>
            <div>
                <p class="text-sm font-bold text-white">${title}</p>
                ${message ? `<p class="text-xs text-slate-400 mt-0.5">${message}</p>` : ''}
            </div>
        </div>
    `;
    
    container.appendChild(note);
    lucide.createIcons();
    
    // Animate in
    setTimeout(() => {
        note.classList.remove('translate-x-10', 'opacity-0');
    }, 10);
    
    // Remove
    setTimeout(() => {
        note.classList.add('translate-x-10', 'opacity-0');
        setTimeout(() => note.remove(), 300);
    }, 4000);
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    switchView('overview');
});
