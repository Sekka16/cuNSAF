# reduce算子优化

## baseline

将数据从global memory读入shared memory，然后相邻元素两两相加，进行规约。、

这样做存在的问题就是产生了warp divergence。对于一个block，所有的thread都是执行同一条命令。

如果存在if-else的分支情况，thread会执行所有的分支，只是产生的结果不会记录下来。

## without warp divergence

解决方法即尽可能地让所有线程走到同一个分支中。

为什么去除了warp divergence之后，不仅性能没有明显的提升，而且精度下降得厉害？
