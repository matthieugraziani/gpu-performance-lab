# 🚀 Benchmark CPU & GPU (PyTorch)

Ce projet fournit des notebooks permettant de mesurer les performances de calcul (en TFLOPS) sur CPU et GPU à l’aide d’opérations de multiplication de matrices avec différentes précisions (float16, float32, float64).

## 📌 Objectifs
- Évaluer les performances réelles du matériel
- Comparer différentes précisions numériques
- Mesurer l’écart avec les performances théoriques
- Visualiser la stabilité et l’efficacité des calculs

## 📂 Structure du projet
~~~bash
.
├── CPU_Benchmark.ipynb   # Benchmark CPU
├── GPU_Benchmark.ipynb   # Benchmark GPU (PyTorch + CUDA)
├── requirements.txt      # Librairies python
└── README.md
~~~

## ⚙️ Méthodologie

Le benchmark repose sur :

- Multiplications de matrices carrées (N x N)
- Plusieurs itérations pour mesurer la stabilité
- Calcul des métriques suivantes :
    - TFLOPS par itération
    - TFLOPS moyen
    - TFLOPS stabilisé
    - TFLOPS peak
  
## 🧪 Précisions testées
- torch.float16 (FP16)
- torch.float32 (FP32 / TF32 sur GPU)
- torch.float64 (FP64)

## 📊 Résultats

Chaque benchmark génère un graphique contenant :

- TFLOPS par itération (barres)
- Moyenne globale
- Moyenne stabilisée
- Peak mesuré
- Pic théorique (si disponible)
- Efficacité (% du pic théorique)

Exemple de sortie :
~~~bash
benchmark_result_float32.png
~~~

## 🧠 Interprétation
- FP16 / FP32 (Tensor Cores) :
Très hautes performances sur GPU moderne
- FP64 :
Performances fortement limitées sur GPU grand public
- Efficacité :
Permet de mesurer l’optimisation réelle vs capacité théorique
## 🖥️ GPU cible (exemple)

Le notebook GPU inclut des références pour des cartes comme :
- RTX 40xx (architecture Ada Lovelace)

Exemple de pics théoriques :
- FP16 / TF32 : ~641 TFLOPS
- FP64 : ~1–2 TFLOPS

## ▶️ Utilisation
1. Installer les dépendances
~~~bash
pip install torch matplotlib
~~~
2. Lancer les notebooks
~~~bash
jupyter notebook
~~~
Puis ouvrir :
~~~bash 
CPU_Benchmark.ipynb 
~~~
~~~bash
GPU_Benchmark.ipynb
~~~

## 📈 Personnalisation

Tu peux modifier :

- Taille des matrices (matrix_size)
- Nombre d’itérations
- Type de données (dtype)
- Activation de TF32 (torch.backends.cuda.matmul.allow_tf32)
  
## ⚠️ Notes
- Les performances dépendent fortement :
    - du matériel
    - de la charge système
    - des optimisations CUDA
- Les premières itérations peuvent être moins représentatives (warmup)
## 📌 Idées d’amélioration
- Benchmark multi-GPU
- Support CPU multi-thread avancé
- Export CSV / JSON des résultats
- Dashboard interactif (Plotly / Streamlit)
## 📜 Licence

Projet libre d’utilisation pour tests et recherche.