#include <RcppArmadillo.h>   
using namespace Rcpp;

// init_L: initialize the L matrix
// tnsr: tensor data, but in the array form
// r1: the rank to compress the 1st mode to
arma::mat init_L(const arma::cube& tnsr, int r1) {
  int a = tnsr.n_rows; // original dimension of mode 1
  int b = tnsr.n_cols; // original dimension of mode 2 
  int n = tnsr.n_slices; // original dimension of mode 3
  
  if (!tnsr.has_nan()){ 
    // If tnsr is complete, use SVD to initialize L
    arma::mat U_mean = arma::zeros<arma::mat>(a, a); // store the U matrices from SVD
    
    for (int i = 0; i < n; i++) {
      arma::mat U, V;
      arma::vec s;
      
      arma::svd(U, s, V, tnsr.slice(i)); // SVD for each slice
      
      U_mean += U; 
    }
    
    U_mean /= n; // Calculate the mean of U matrices from all the slices
    
    return U_mean.cols(0,r1-1); // Let L be the top r1 columns of U
    
  } else {
    // If tnsr is not complete, initialize L with (I_{r1}, 0)^T
    arma::mat L0(a, r1, arma::fill::zeros);  
    L0.submat(0, 0, r1 - 1, r1 - 1) = arma::eye(r1, r1);  
    return L0;
    
  }
}

arma::cube ImputeTnsr_cpp(const arma::cube& tnsr, double num){
  arma::cube imputed = tnsr;
  imputed.elem(find_nonfinite(tnsr)).fill(num);
  return imputed;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List cglram(const arma::cube& tnsr, const arma::vec& ranks, double lambda, 
                  Rcpp::Nullable<arma::mat> L0_, const arma::mat& D, double tol, int max_iter) {
  int a = tnsr.n_rows;
  int b = tnsr.n_cols;
  int n = tnsr.n_slices;
  
  int r1 = ranks(0);
  int r2 = ranks(1);
  
  arma::mat L0;
  if (L0_.isNull()) {
    L0 = init_L(tnsr, r1);
  } else {
    L0 = Rcpp::as<arma::mat>(L0_);
  }
  
  arma::mat A = arma::eye(D.n_cols, D.n_cols) + lambda * D.t() * D;
  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, A);
  
  arma::mat Q_inv_sqrt = arma::diagmat(1 / arma::sqrt(eigval));
  arma::mat A_inv_sqrt = eigvec * Q_inv_sqrt * eigvec.t();
  
  arma::mat Q_sqrt = arma::diagmat(arma::sqrt(eigval));
  arma::mat A_sqrt = eigvec * Q_sqrt * eigvec.t();
  
  arma::mat X = A_sqrt * L0;
  arma::mat U, V;
  arma::vec s;
  arma::svd(U, s, V, X);
  
  arma::mat R, L;
  arma::cube G(r1, r2, n, arma::fill::zeros);
  arma::cube est(a, b, n, arma::fill::zeros);
  arma::vec obj_func(max_iter, arma::fill::zeros);
  
  bool converged = false;
  int curr_iter = 0;
  
  while ((curr_iter < max_iter) && (!converged)) {
    arma::mat M_R(b, b, arma::fill::zeros);
    for (int i = 0; i < n; i++) {
      arma::mat M_i = tnsr.slice(i);
      arma::mat half_M_R = M_i.t() * A_inv_sqrt * U;
      M_R += half_M_R * half_M_R.t();
    }
    arma::eig_sym(eigval, eigvec, M_R);
    arma::uvec indices_R = arma::sort_index(eigval, "descend"); 
    eigvec = eigvec.cols(indices_R);  
    R = eigvec.head_cols(r2);
    
    arma::mat M_U(a, a, arma::fill::zeros);
    for (int i = 0; i < n; i++) {
      arma::mat M_i = tnsr.slice(i);
      arma::mat half_M_U = A_inv_sqrt * M_i * R;
      M_U += half_M_U * half_M_U.t();
    }
    arma::eig_sym(eigval, eigvec, M_U);
    arma::uvec indices_U = arma::sort_index(eigval, "descend"); 
    eigvec = eigvec.cols(indices_U);  
    U = eigvec.head_cols(r1);
    
    arma::mat cal_L = A_inv_sqrt * U * U.t() * A_inv_sqrt;
    arma::vec eigval_cal_L;
    arma::mat eigvec_cal_L;
    arma::eig_sym(eigval_cal_L, eigvec_cal_L, cal_L);
    arma::uvec indices_L = arma::sort_index(eigval_cal_L, "descend"); 
    eigvec_cal_L = eigvec_cal_L.cols(indices_U);
    L = eigvec_cal_L.head_cols(r1);
    
    
    double f_sum = 0.0;
    for (int i = 0; i < n; i++) {
      arma::mat M_i = tnsr.slice(i);
      
      arma::mat LTMR = L.t() * M_i * R;
      arma::mat prior_term = arma::eye(r1, r1) + lambda * (L.t() * D.t() * D * L);
      arma::mat prior_term_inv = arma::inv(prior_term);
      
      G.slice(i) = prior_term_inv * LTMR;
      est.slice(i) = L * G.slice(i) * R.t();
      
      arma::mat term1 = M_i - est.slice(i);
      arma::mat term2 = D * est.slice(i);
      
      f_sum += arma::norm(term1, "fro") * arma::norm(term1, "fro") +
        lambda * arma::norm(term2, "fro") * arma::norm(term2, "fro");
    }
    obj_func(curr_iter) = f_sum;
    
    if (curr_iter > 0 && std::abs(obj_func(curr_iter) - obj_func(curr_iter - 1)) < tol) {
      converged = true;
    }
    
    if (!converged) {
      curr_iter++;
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("L") = L,
    Rcpp::Named("R") = R,
    Rcpp::Named("G") = G,
    Rcpp::Named("est") = est,
    Rcpp::Named("conv") = converged,
    Rcpp::Named("obj_func") = obj_func.subvec(0, curr_iter - 1)
  );
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List mglram(const arma::cube& tnsr, const arma::vec& ranks, double lambda, 
                      Rcpp::Nullable<arma::mat> L0_, const arma::mat& D, double tol=1e-5, int max_iter=500, double init=0) {
  int a = tnsr.n_rows;
  int b = tnsr.n_cols;
  int n = tnsr.n_slices;
  
  int r1 = ranks(0);
  int r2 = ranks(1);
  
  arma::mat L0;
  if (L0_.isNull()) {
    L0 = init_L(tnsr, r1);
  } else {
    L0 = Rcpp::as<arma::mat>(L0_);
  }

  arma::cube filled_tnsr = tnsr;
  arma::cube indicator(a, b, n, arma::fill::ones);
  if (!tnsr.has_nan()){
    return cglram(tnsr, ranks, lambda, Rcpp::wrap(L0), D, tol, max_iter);
  }
  
  if (tnsr.has_nan()){
    indicator.elem(arma::find_nonfinite(tnsr)).fill(0);
    filled_tnsr = ImputeTnsr_cpp(tnsr, init);
  }
  
  bool converged = false;
  int curr_iter = 0;
  arma::mat L, R;
  arma::cube G(r1, r2, n, arma::fill::zeros);
  arma::cube est(a, b, n, arma::fill::zeros);
  arma::cube new_M(a, b, n, arma::fill::zeros);
  arma::vec obj_func(max_iter, arma::fill::zeros);

  while ((curr_iter < max_iter) && (!converged)) {
    Rcpp::List glram_res = cglram(filled_tnsr, ranks, lambda, Rcpp::wrap(L0), D, tol, max_iter);
    
    L = Rcpp::as<arma::mat>(glram_res["L"]);
    R = Rcpp::as<arma::mat>(glram_res["R"]);
    G = Rcpp::as<arma::cube>(glram_res["G"]);
    est = Rcpp::as<arma::cube>(glram_res["est"]);

    double term1 = 0;
    double term2 = 0;

    for (int i=0; i < n; i++){
      arma::mat M_i = filled_tnsr.slice(i);
      arma::mat H_i = indicator.slice(i);
      arma::mat G_i = G.slice(i);
      arma::mat est_i = est.slice(i);
      arma::mat mat1(a, b, arma::fill::ones);
      new_M.slice(i) = est_i % (mat1-H_i) + M_i % H_i;

      term1 += std::pow(arma::norm((M_i % H_i) - (est_i % H_i), "fro"), 2);
      arma::mat term2_mat = D * L * G_i * R.t();
      term2 += lambda * std::pow(arma::norm(term2_mat, "fro"), 2);

    }
    obj_func(curr_iter) = term1 + term2;

    if (curr_iter > 0 && std::abs(obj_func(curr_iter) - obj_func(curr_iter - 1)) < tol) {
      converged = true;
    }

    if (!converged) {
      curr_iter++;
      filled_tnsr = new_M;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("L") = L,
    Rcpp::Named("R") = R,
    Rcpp::Named("G") = G,
    Rcpp::Named("est") = est,
    Rcpp::Named("filled_tnsr") = filled_tnsr,
    Rcpp::Named("conv") = converged,
    Rcpp::Named("obj_func") = obj_func.subvec(0, curr_iter - 1)
  );
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List oracle(const arma::cube& tnsr, const arma::cube& smooth_tnsr, const arma::mat& rank_grid, const arma::vec& lambda_seq,
                  Rcpp::Nullable<arma::mat> L0_, const arma::mat& D, double tol=0.1, int max_iter=500, double init=0){
  int n_ranks = rank_grid.n_rows;
  int n_lambda = lambda_seq.n_elem;
  int n = smooth_tnsr.n_elem;
  arma::mat error(n_ranks, n_lambda, arma::fill::zeros);

  arma::vec rank_i;
  double lambda_j;
  for (int i = 0; i < n_ranks; i++){
    rank_i = rank_grid.row(i).t();
    for (int j = 0; j < n_lambda; j++){
      lambda_j = lambda_seq(j);
      Rcpp::List res = mglram(tnsr, rank_i, lambda_j, L0_, D, tol, max_iter, init);
      arma::cube est = Rcpp::as<arma::cube>(res["est"]);
      arma::mat L0_ = Rcpp::as<arma::mat>(res["L"]);

      arma::cube diff = est - smooth_tnsr;
      error(i,j) = arma::accu(diff % diff) / n;
    }
  }
  arma::uword min_index = error.index_min();

  arma::uword row = min_index % error.n_rows;
  arma::uword col = min_index / error.n_rows;
  
  arma::rowvec opt_para = arma::join_horiz(rank_grid.row(row).cols(0, 1), arma::rowvec({lambda_seq(col)}));

  return Rcpp::List::create(
    Rcpp::Named("error") = error,
    Rcpp::Named("opt_para") = opt_para
  );
}  


arma::mat LambdaSeqFit(const arma::cube& tnsr, const arma::vec& ranks, const arma::vec& lambda_seq,
                       Rcpp::Nullable<arma::mat> L0_, const arma::mat& D, double tol, int max_iter, double init){
  int n_lambda = lambda_seq.n_elem;

  int a = tnsr.n_rows;
  int b = tnsr.n_cols;
  int n = tnsr.n_slices;

  int r1 = ranks(0);
  arma::mat L0_update = init_L(tnsr, r1);

  arma::mat vectorized_tnsrs(a*b*n, n_lambda, arma::fill::zeros);
  for (int i = 0; i < n_lambda; i++){
    double lambda_i = lambda_seq(i);
    Rcpp::List res = mglram(tnsr, ranks, lambda_i, Rcpp::wrap(L0_update), D, tol, max_iter, init);
    arma::cube est = Rcpp::as<arma::cube>(res["est"]);
    arma::mat L = Rcpp::as<arma::mat>(res["L"]);
    L0_update = L;
    arma::vec vec_tnsr = arma::vectorise(est);
    vectorized_tnsrs.col(i) = vec_tnsr;

  }
  return vectorized_tnsrs;
}

arma::vec grouping(const arma::cube& tnsr, int k){
  arma::uvec nmiss_idx = arma::find_finite(tnsr);
  int data_size = nmiss_idx.n_elem;
  arma::vec vec = arma::linspace<arma::vec>(1, data_size, data_size);
  arma::vec group_unshuffle(data_size);
  for (int i = 0; i < data_size; i++){
    group_unshuffle(i) = std::fmod(vec(i), k) + 1;
  }
  arma::vec group_shuffle = arma::shuffle(group_unshuffle);
  return group_shuffle;
}


Rcpp::List kFoldLambda(const arma::cube& tnsr, const arma::vec& ranks, const arma::vec& lambda_seq, int k,
                      Rcpp::Nullable<arma::mat> L0_, const arma::mat& D, double tol, int max_iter, double init){
  int n_lambda = lambda_seq.n_elem;
  arma::uvec nmiss_idx = arma::find_finite(tnsr);
  arma::vec groups = grouping(tnsr, k);
  arma::vec vec_tnsr = arma::vectorise(tnsr);
  
  arma::mat MSE(n_lambda, k, arma::fill::zeros);
  for (int i = 1; i < k+1; i++){
    arma::uvec masked_idx = nmiss_idx(find(groups == i));
    arma::cube masked_tnsr = tnsr;  
    masked_tnsr.elem(masked_idx).fill(arma::datum::nan);  
    
    arma::mat k_res = LambdaSeqFit(masked_tnsr, ranks, lambda_seq, L0_, D, tol, max_iter, init);
    
    arma::vec masked_values = vec_tnsr.elem(masked_idx);
    arma::mat est_values = k_res.rows(masked_idx);
    
    arma::rowvec squared_errors = sum(square(est_values.each_col() - masked_values), 0)/(masked_idx.n_elem);
    MSE.col(i-1) = squared_errors.t();
  }
  
  
  arma::vec MSE_vec = mean(MSE, 1);
  arma::vec SE_vec = stddev(MSE, 0, 1)/std::pow(k,0.5);
 
  return Rcpp::List::create(
    Rcpp::Named("MSE_vec") = MSE_vec,
    Rcpp::Named("SE_vec") = SE_vec
  );
  
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List kcv(const arma::cube& tnsr, const arma::mat& rank_grid, const arma::vec& lambda_seq, int k,
               Rcpp::Nullable<arma::mat> L0_, const arma::mat& D, double tol=0.1, int max_iter=500, double init=0){
  int n_ranks = rank_grid.n_rows;
  int n_lambda = lambda_seq.n_elem;
  
  arma::mat MSE_mat(n_ranks, n_lambda, arma::fill::zeros);
  arma::mat SE_mat(n_ranks, n_lambda, arma::fill::zeros);
  arma::vec lambda_SE_vec(n_lambda, arma::fill::zeros);
  
  for (int i = 0; i < n_ranks; i++){
    arma::vec rank_i = rank_grid.row(i).t();
    Rcpp::List lambda_res = kFoldLambda(tnsr, rank_i, lambda_seq, k, L0_, D, tol, max_iter, init);
    
    arma::vec lambda_MSE_vec = Rcpp::as<arma::vec>(lambda_res["MSE_vec"]);
    arma::vec lambda_SE_vec = Rcpp::as<arma::vec>(lambda_res["SE_vec"]);
    
    MSE_mat.row(i) = lambda_MSE_vec.t();
    SE_mat.row(i) = lambda_SE_vec.t();
    
  }
  
  arma::uword min_index = MSE_mat.index_min();
  
  arma::uword row = min_index % MSE_mat.n_rows;
  arma::uword col = min_index / MSE_mat.n_rows;
  
  arma::rowvec opt_para = arma::join_horiz(rank_grid.row(row).cols(0, 1), arma::rowvec({lambda_seq(col)}));
  
  return Rcpp::List::create(
    Rcpp::Named("MSE_mat") = MSE_mat,
    Rcpp::Named("SE_mat") = SE_mat, 
    Rcpp::Named("opt_para") = opt_para
  );
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List loss(const arma::cube& tnsr, const arma::cube& smooth_tnsr, 
                const arma::mat& L, const arma::mat& true_L, 
                const arma::mat& R, const arma::mat& true_R){
  int n = tnsr.n_elem;
  
  arma::cube diff_M = tnsr - smooth_tnsr;
  double loss_M = arma::accu(diff_M % diff_M) / n;
  
  arma::mat diff_L = true_L * true_L.t() - L * L.t();
  double loss_L = std::pow(0.5, 0.5) * std::pow(arma::accu(diff_L % diff_L), 0.5);
  
  arma::mat diff_R = true_R * true_R.t() - R * R.t();
  double loss_R = std::pow(0.5, 0.5) * std::pow(arma::accu(diff_R % diff_R), 0.5);
  
  return Rcpp::List::create(
    Rcpp::Named("loss_M") = loss_M,
    Rcpp::Named("loss_L") = loss_L,
    Rcpp::Named("loss_R") = loss_R
  );
}




