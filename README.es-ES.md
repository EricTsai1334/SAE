

# SAE: Similarity Aware Evaluation (ICLR 2025 Oral)

Código para el artículo SAE: Rethinking the Generalization of Drug Target Affinity Prediction Algorithms via Similarity Aware Evaluation.
> [Rethinking the Generalization of Drug Target Affinity Prediction Algorithms via Similarity Aware Evaluation](https://openreview.net/forum?id=j7cyANIAxV)  \
> Autores: Chenbin Zhang, Zhiqiang Hu, Chuchu Jiang, Wen Chen, Jie Xu, Shaoting Zhang

Contacto: chenbinzhang@moleculemind.com o huzq@pku.edu.cn. ¡No duden en hacer preguntas o iniciar discusiones!

## Introducción
La Evaluación Consciente de la Similitud (SAE) es una nueva metodología de división de conjuntos de entrenamiento y prueba que puede lograr una distribución de similitud deseada mediante descenso de gradiente.

## Requisitos
```bash
python >= 3.7
torch >= 1.11
rdkit >= 2023.3.1
pandas
numpy
seaborn
matplotlib
omegaconf
```

Para reproducir completamente nuestros resultados de división, podemos configurar el entorno de la siguiente manera:
```bash
python==3.7.16
torch==1.11.0+cu113
numpy==1.21.6
rdkit==2023.3.1
```
Y los paquetes restantes no requieren versiones específicas.

## Inicio rápido
Puede dividir el conjunto de datos en conjuntos de entrenamiento y prueba ejecutando el siguiente comando:

```bash
python SAE_split.py demo_configs/IC50_EGFR_balance.yaml
```

Después de ejecutar el comando, los archivos de salida se organizarán de la siguiente manera:

```bash
demo_output/
    └── IC50_EGFR
        ├── info.csv
        └── sigmoid-True_base-lr-1.00e-02_optim-kind-ExtraAdam_sched-kind-CosineAnnealing_init-kind-custom_lamb-2.03e-03_sigma-0.10_scale-factor-100_bins-89e6b15907a9354ec4caf6e18ce378db_seed-233_init-scale-5_max-iters-20000
            ├── W.npy
            ├── real_R.npy
            ├── sim.png
            ├── test.csv
            ├── train.csv
            ├── viz_W.png
            └── viz_real_R.png
```

### Explicación de los archivos de salida:
- **info.csv**: Contiene metadatos o información adicional sobre la división del conjunto de datos.
- **W.npy**: La matriz de pesos resultante.
- **real_R.npy**: La matriz de valores reales generada durante la división.
- **test.csv**: El conjunto de prueba resultante.
- **train.csv**: El conjunto de entrenamiento resultante.
- **viz_W.png**: Una representación visual de la matriz de pesos.
- **viz_real_R.png**: Una representación visual de la matriz de similitud.

## Búsqueda en cuadrícula (Grid-search) para la división de datos

El archivo de configuración de demostración es `demo_configs/IC50_EGFR_mimic.yaml`, el cual está organizado de la siguiente manera: (Los comentarios explican los valores opcionales de los hiperparámetros.)
```yaml
dataset_path: $input_csv_path
save_dir: $save_dir
test_ratio: 0.2
hyper_param_dict:
    'bins': [ ]
    'seed': [233, ]
    'max_iters': [20000, ]
    'sigmoid': [True, ]  # {True, False}
    'base_lr': [1e-2, ]
    'optim_kind': ['ExtraAdam', ]  # {ExtraSGD, ExtraAdam, SGD, Adam, AdamW}
    'sched_kind': ['CosineAnnealing', ]  # {CosineAnnealing, CAWarmRStarts, Step}
    'init_kind': ['custom', ]  # {normal, uniform, custom}
    'lamb': [2.03091762e-03, ]
    'sigma': [0.1, ]
    'scale_factor': [100, ]
    'init_scale': [5, ]
```
El parámetro `bins` tiene dos opciones de configuración:

1. **Sin asignar pesos a cada intervalo (bin)**: Por ejemplo, para lograr una división equilibrada como se describe en el artículo, como "[0, 1/3, 2/3, 1]", podemos configurar `bins` como `'bins': [[0, 1 / 3, 2 / 3, 1.0], ]` o `'bins': [[0, 0.33333, 0.66666, 1.0], ]`. Para una división "0.4-0.6", donde se espera que la distribución de prueba tenga su máxima similitud en el rango entre 0.4 y 0.6, podemos configurar `bins` como `'bins': [[0.4, 0.6], ]`.

2. **Con asignación de pesos a cada intervalo (bin)**: Para una división de imitación (mimic) como se discute en el artículo, primero calculamos la cantidad de elementos por intervalo del conjunto de prueba externo. Luego, podemos configurar `bins` de la siguiente manera: (`SAE_split.py` solo se centra en los tamaños relativos de estas cantidades de intervalos, por lo que no requiere que la suma de las cantidades coincida exactamente con el tamaño del conjunto de prueba.)
    ```yaml
    'bins': [
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0, 6, 36, 119, 247, 460, 271, 129, 52, 12]
        ],
    ]
    ```

## Otros scripts
1. `select_split.py`
    - Descripción: selecciona el resultado de división óptimo en el experimento de búsqueda en cuadrícula.
    - Uso: `python select_split.py ${path to info.csv in the output directory of grid-search experiment}`
2. `double_check_split.py`
    - Descripción: verifica nuevamente la distribución de similitud entre entrenamiento y prueba.
    - Uso: `python double_check_split.py $dataset_dir $train.csv $test.csv $save_path`
    - Ejemplo: 
    ```bash
    python double_check_split.py demo_output/IC50_EGFR/sigmoid-True_base-lr-1.00e-02_optim-kind-ExtraAdam_sched-kind-CosineAnnealing_init-kind-custom_lamb-2.03e-03_sigma-0.10_scale-factor-100_bins-89e6b15907a9354ec4caf6e18ce378db_seed-233_init-scale-5_max-iters-20000/ train.csv test.csv ./viz.jpg
    ```


## Cita
Si utiliza este código en su investigación, por favor cite el siguiente artículo:

```bibtex
@inproceedings{zhang2025rethinking,
    title={Rethinking the generalization of drug target affinity prediction algorithms via similarity aware evaluation},
    author={Chenbin Zhang and Zhiqiang Hu and Jiang Chuchu and Wen Chen and JIE XU and Shaoting Zhang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=j7cyANIAxV}
}
```
