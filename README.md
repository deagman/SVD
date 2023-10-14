## 赛题附加说明

>附件：
>
>depend: rocSOLVER以及测试程序的依赖包目录
>
>rocSOLVER-rocm-5.4.3.tar.gz：待优化rocSOLVER源码包
>
>test：测试程序目录

### rocSOLVER编译流程

```shell
tar -xzvf rocSOLVER-rocm-5.4.3.tar.gz
cd rocSOLVER-rocm-5.4.3/build
sbatch make.slurm
```

- make.slurm中可根据账号的队列替换队列名称

```bash
#SBATCH -p queue_name
```

- 编译输出可查看job.out、job.err文件
- 首次编译过程大约持续20分钟，可通过squeue命令查看任务状态，编译任务完成后进入`rocSOLVER-rocm-5.4.3/build/release/rocsolver-install`目录查看编译输出的库文件
- 优化代码需要集成到`rocSOLVER-rocm-5.4.3/library`目录下，然后再进行编译测试

### 测试验证

```shell
cd test
source env.sh
make
#确认测试程序是否链接到本地rocSOLVER库
ldd -r gesvd_test 
#测试程序使用方法如下：
sbatch run.slurm [m] [n] [function] [test] [iters] 
# 参数说明
# m: 矩阵行数（必填）
# n: 矩阵列数（必填）
# function: （选填，默认为1）
#     1——rocsolver_dgesvd 
#     2——rocsolver_zgesvd
# test: （可选，Test1和Test2验证方法在赛题文档中有具体说明，默认为2，使用Test2方法验证）
#     0——不进行正确性验证
#     1——使用Test1方法验证
#     2——使用Test2方法验证
#     3——使用Test1和Test2方法验证
# iters: 测试时长接口调用次数，取平均值。（可选，默认10次）
#
# example: sbatch run.slurm 100 90 1 2 10 
```

