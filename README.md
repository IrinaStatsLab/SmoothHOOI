
<!-- README.md is generated from README.Rmd. Please edit that file -->

# SmoothHOOI

<!-- badges: start -->

<!-- badges: end -->

SmoothHOOI is an R package that implements methods described in ‘’Smooth
Tensor Decomposition with Application to Ambulatory Blood Pressure
Monitoring Data’’

## Installation

You can install the development version of SmoothHOOI from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("IrinaStatsLab/SmoothHOOI")
```

## Example

This is a simple example which shows you how to use this package.

``` r
library(SmoothHOOI)

set.seed(1234321)

# Generate a random tensor with missing data, in the array format
dims <- c(24, 3, 207)
tnsr <- array(rnorm(prod(dims), mean = 0, sd = 1), dim = dims)

missing_prob <- 0.2
missing_indices <- sample(length(tnsr), size = floor(length(tnsr) * missing_prob))
tnsr[missing_indices] <- NA
```

``` r
# Make the second order difference matrix with circular nature
D2 <- SecDiffMat(24)
```

``` r
# Find optimal hyperparameter with 5-fold cross-validation
kcv_res <- kcv(tnsr, rank_grid=as.matrix(expand.grid(r1<-seq(3,4,by=1), r2<-c(2,3))), lambda_seq=seq(1,10,by=1), k=5, L0=NULL, D=D2, tol=0.01, max_iter=500, init=0)

kcv_res$MSE_mat # matrix for CV error (rows representing ranks, cols representing lambda)
#>          [,1]     [,2]     [,3]     [,4]     [,5]     [,6]     [,7]     [,8]
#> [1,] 1.172131 1.166626 1.161576 1.158826 1.156901 1.155217 1.153676 1.152170
#> [2,] 1.227177 1.205206 1.190116 1.181757 1.175460 1.170243 1.165838 1.162020
#> [3,] 1.257812 1.257523 1.252646 1.249176 1.246047 1.243126 1.240320 1.237611
#> [4,] 1.339317 1.310459 1.292404 1.279279 1.268859 1.260312 1.253015 1.246766
#>          [,9]    [,10]
#> [1,] 1.150657 1.149116
#> [2,] 1.158628 1.155584
#> [3,] 1.235024 1.232384
#> [4,] 1.241243 1.236316
kcv_res$opt_para # optimal hyperparameters
#>      [,1] [,2] [,3]
#> [1,]    3    2   10
```

``` r
# Run SmoothHOOI algorithm with the optimal hyperparameters
res <- mglram(tnsr, ranks = c(4, 2), init=0, D = D2,
       lambda = 20, max_iter = 500, tol = 1e-5, L0 = NULL)

res$conv # check convergence
#> [1] TRUE
```

``` r
# Rotation for Identifiability
tilde <- MakeIdent(L=res$L, G=res$G, R=res$R)

tilde$L_tilde
#>            [,1]        [,2]         [,3]        [,4]
#>  [1,] 0.1840606  0.22393244 -0.213218932  0.28042853
#>  [2,] 0.1646762  0.26624741 -0.147708965  0.27210654
#>  [3,] 0.1482340  0.29660216 -0.070726062  0.18816304
#>  [4,] 0.1374378  0.31349852  0.009615428  0.04939753
#>  [5,] 0.1323590  0.31271460  0.087700010 -0.10376905
#>  [6,] 0.1342206  0.29326459  0.159545671 -0.22997081
#>  [7,] 0.1418485  0.25561940  0.218194235 -0.28923744
#>  [8,] 0.1534213  0.20216557  0.259996865 -0.26244835
#>  [9,] 0.1681931  0.13465631  0.283399728 -0.16237717
#> [10,] 0.1847660  0.05704478  0.288284636 -0.01725593
#> [11,] 0.2011746 -0.02271733  0.276560176  0.13409177
#> [12,] 0.2153695 -0.09544272  0.251438678  0.24738159
#> [13,] 0.2271149 -0.15733050  0.214691974  0.29191150
#> [14,] 0.2344114 -0.20461454  0.169171445  0.25559853
#> [15,] 0.2396923 -0.23553426  0.113451881  0.15520290
#> [16,] 0.2438257 -0.24808355  0.048534854  0.01568602
#> [17,] 0.2471621 -0.24162042 -0.021994027 -0.12538837
#> [18,] 0.2493763 -0.21525308 -0.094212255 -0.23273723
#> [19,] 0.2494586 -0.17023044 -0.163827192 -0.28415778
#> [20,] 0.2467092 -0.10973835 -0.223196354 -0.26866852
#> [21,] 0.2402318 -0.03974661 -0.265093128 -0.18600678
#> [22,] 0.2304089  0.03311546 -0.285959885 -0.05696033
#> [23,] 0.2185785  0.10467018 -0.283949427  0.08704279
#> [24,] 0.2031827  0.16936512 -0.259831875  0.21053358
tilde$R_tilde
#>            [,1]       [,2]
#> [1,] -0.5448117  0.8085874
#> [2,]  0.1760055 -0.1487962
#> [3,]  0.8198794  0.5692505
tilde$G_tilde[ , ,1]
#>            [,1]        [,2]
#> [1,] -0.2287400  2.20037009
#> [2,]  0.6527040  0.73184720
#> [3,]  0.2652866  0.04476627
#> [4,] -0.3216800 -0.27923386
```
