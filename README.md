![](http://hertzcat.com/2018/03/24/coursera-ml-andrewng-linear-regression/coursera-ml-andrewng-cover.jpeg)
项目简介
------------
![](https://img.shields.io/pypi/pyversions/Django.svg)

本项目是 吴恩达机器学习 课程的 笔记 和 作业。课程作业原先使用的是 Octave 和 MATLAB。不过笔者觉得无论是对 Machine Learning 的学习还是对未来工程项目的开发 Python 都更为合适。所以笔者就使用 Python 将课程作业重新实现了一遍。
希望这个项目能帮助大家理清课程的内容，理解算法背后的模型，掌握一些 Python 基本库的使用。

Python 依赖包
------------
```bash
pip install numpy
pip install matplotlib
pip install scipy
```

第一周 | 线性回归
------------
* [笔记](http://hertzcat.com/2018/03/24/coursera-ml-andrewng-linear-regression/)
* [作业说明](https://github.com/hertzcat/Coursera-ML-AndrewNg-Python/blob/master/ml-ex1/ex1.pdf)
* 数据：`ex1data1.txt`，`ex1data2.txt`
* 作业文件：`ex1.py`，`ex1_multi.py`

```bash
python ex1.py 
python ex1_multi.py 
```

第二周 | 逻辑回归
------------
* [笔记](http://hertzcat.com/2018/03/31/coursera-ml-andrewng-logistic-regression/)
* [作业说明](https://github.com/hertzcat/Coursera-ML-AndrewNg-Python/blob/master/ml-ex2/ex2.pdf)
* 数据：`ex2data2.txt`，`ex2data2.txt`
* 作业文件：`ex2.py`，`ex2_reg.py`

```bash
python ex2.py 
python ex2_reg.py 
```

第三周 | 神经网络 | 多分类问题
------------
* [笔记](http://hertzcat.com/2018/04/07/coursera-ml-andrewng-nn-multi-class/)
* [作业说明](https://github.com/hertzcat/Coursera-Machine-Learning/blob/master/ml-ex3/ex3.pdf)
* 数据：`ex3data1.mat`，`ex3weights.mat`
* 作业文件：`ex3.py`，`ex3_nn.py`

```bash
python ex3.py 
python ex3_nn.py 
```

第四周 | 神经网络 | 反向传播算法
------------
* [笔记](http://hertzcat.com/2018/04/14/coursera-ml-andrewng-nn-back-propagation/)
* [作业说明](https://github.com/hertzcat/Coursera-Machine-Learning/blob/master/ml-ex4/ex4.pdf)
* 数据：`ex4data1.mat`，`ex4weights.mat`
* 作业文件：`ex4.py`

```bash
python ex4.py
```

第五周 | 方差与偏差
------------
* [笔记](http://hertzcat.com/2018/04/21/coursera-ml-andrewng-bias-vs-variance/)
* [作业说明](https://github.com/hertzcat/Coursera-Machine-Learning/blob/master/ml-ex5/ex5.pdf)
* 数据：`ex5data1.mat`
* 作业文件：`ex5.py`

```bash
python ex5.py
```

第六周 | 支持向量机
------------
* [笔记](http://hertzcat.com/2018/05/13/coursera-ml-andrewng-svm/)
* [作业说明](https://github.com/hertzcat/Coursera-Machine-Learning/blob/master/ml-ex6/ex6.pdf)
* [cs229 讲义](https://github.com/hertzcat/Coursera-Machine-Learning/blob/master/ml-ex6/cs229-notes3.pdf)
* [cs299 SMO](https://github.com/hertzcat/Coursera-Machine-Learning/blob/master/ml-ex6/smo.pdf)
* [SMO 论文](https://github.com/hertzcat/Coursera-Machine-Learning/blob/master/ml-ex6/smo-book.pdf)
* 数据：`ex6data1.mat`，`ex6data2.mat`，`ex6data3.mat`，`spamTrain.mat`，`spamTest.mat` ...
* 作业文件：`ex6.py`，`ex6_spam.py`

```bash
python ex6.py
python ex6_spam.py
```

第七周 | 无监督学习算法 | k-means 与 PCA
------------
* [笔记](http://hertzcat.com/2018/06/05/coursera-ml-andrewng-kmeans-and-pca/)
* [作业说明](https://github.com/hertzcat/Coursera-Machine-Learning/blob/master/ml-ex7/ex7.pdf)
* 数据：`bird_small.png`，`ex7data1.mat`，`ex7data2.mat`，`ex7faces.mat`
* 作业文件：`ex7.py`，`ex7_pca.py`

```bash
python ex7.py
python ex7_pca.py
```

第八周 | 异常检测与协同过滤
------------
* [笔记](http://hertzcat.com/2018/07/07/coursera-ml-andrewng-anomaly-detection-and-collaborative-filtering/)
* [作业说明](https://github.com/hertzcat/Coursera-Machine-Learning/blob/master/ml-ex8/ex8.pdf)
* 数据：`ex8data1.mat`，`ex8data2.mat`，`ex8_movies.mat`，`ex8_movieParams.mat`，`movie_ids.txt`
* 作业文件：`ex8.py`，`ex8_cofi.py`

```bash
python ex8.py
python ex8_cofi.py
```