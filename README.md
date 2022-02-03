# Multi-Task_Hierarchical_Learning
This is the implementation of the paper ["Multi-Task Hierarchical Learning Based Network Traffic Analytics" (ICC), 2020](https://ieeexplore.ieee.org/abstract/document/9500546)

## Dataset

The dataset splits are NetML, CICIDS2017, and ISCX-VPN-nonVPN2016 which are curated from [NetML Competition 2020](https://github.com/ACANETS/NetML-Competition2020). Three datasets are provided under data/ folder.

## Requirements

- python 3.5.8 or higher
- matplotlib
- tensorflow-gpu 2.3.2 or higher
- pandas
- scikit-learn 

## Create and setup the virtual Environment

```shell
python3 -m venv ./env
```

```shell
source ./env/bin/activate
```

```shell
pip install -r requirements.txt
```

## Baseline Training and Testing

main.py iteratively trains and tests all the datasets and model combinations. The results are saved under ./results/ folder. When the virtual environment is activated:


```shell
python3 main.py
```

## MTHL (Multi-Task Hierarchical Learning) Model Training and Testing

When the virtual environment is activated:

```shell
python3 multi_label.py 
```

## References

If this repository was useful for your research, please cite our paper:

```
@INPROCEEDINGS{9500546,  author={Barut, Onur and Luo, Yan and Zhang, Tong and Li, Weigang and Li, Peilong},  booktitle={ICC 2021 - IEEE International Conference on Communications},   title={Multi-Task Hierarchical Learning Based Network Traffic Analytics},   year={2021},  volume={},  number={},  pages={1-6},  doi={10.1109/ICC42927.2021.9500546}}
```