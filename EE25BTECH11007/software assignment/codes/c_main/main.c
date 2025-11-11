#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" 

#define TOL        1e-8
#define MAX_SWEEPS 100

typedef struct {
    int rows, cols;
    double *d;
} MATRIX;

// Flat array coords
int idx(int i, int j, int cols)
{ return i*cols + j; }

// memory allocation
MATRIX mat_alloc(int m, int n){
    MATRIX A; A.rows=m; A.cols=n;
    size_t bytes = (size_t)m * (size_t)n * sizeof(double);
    A.d = (double*)malloc(bytes);
    return A;
}

void mat_free(MATRIX *A) {
     free(A->d); A->d = NULL;
     }

void mat_identity(MATRIX A){
    int m=A.rows, n=A.cols; 
    int r=(m<n?m:n);
    if(m<n)r=m;
    else r=n;
    for (int k=0;k<m*n;k++) A.d[k]=0.0;
    for (int i=0;i<r;++i){
        A.d[idx(i,i,n)]=1.0;
    }   
}

// pgm_write
int pgm_write(const char *path, MATRIX A)
{
    // 1) Open file in binary mode
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        perror("open");
        return -1;
    }

    // 2) Minimal PGM (P5) header: magic, width height, maxval
    if (fprintf(fp, "P5\n%d %d\n255\n", A.cols, A.rows) < 0) {
        fclose(fp);
        return -1;
    }

    // 3) Stream pixels: clamp to [0,255]
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            double v = A.d[idx(i, j, A.cols)];

            // clamp
            if (v < 0.0) v = 0.0;
            else if (v > 255.0) v = 255.0;

            // round and emit
            unsigned char b = (unsigned char)(v + 0.5);
            fputc((int)b, fp);
        }
    }

    fclose(fp);
    return 0;
}

// Loads PJEG/PNG in stb_img.h
int stb_read_gray_matrix(const char *path, MATRIX *Aout)
{
    int w = 0, h = 0, comps_in = 0;

    //RGB to Greyscale
    unsigned char *img = stbi_load(path, &w, &h, &comps_in, 1);

    //Failure
    if (img == NULL) {
        fprintf(stderr, "stb_image: failed to load '%s' (%s)\n",
                path, stbi_failure_reason());
        return -1;
    }

    // Destination Matrix
    MATRIX A = mat_alloc(h, w);
    if (A.d == NULL) {
        stbi_image_free(img);
        return -1;
    }

    // Copy flat array coords
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            A.d[idx(i, j, w)] = (double)img[i * w + j];
        }
    }
    // Free stb buffer
    stbi_image_free(img);
    *Aout = A;
    return 0;
}

// dot prod
double col_dot(MATRIX M, int p, int q){
    double s=0.0; int m=M.rows, n=M.cols;
    for (int i=0;i<m;i++){
        s += M.d[idx(i,p,n)] * M.d[idx(i,q,n)];
    } 
    return s;
}

// norm
double col_norm(MATRIX M, int j){
    double s2=0.0; int m=M.rows, n=M.cols;
    for (int i=0;i<m;i++){
         double v=M.d[idx(i,j,n)]; s2+=v*v;
        }
    return sqrt(s2);
}

// updated columns
void rotate_cols(MATRIX M, int p, int q, double c, double s){
    int m=M.rows, n=M.cols;
    for (int i=0;i<m;i++){
        double ap = M.d[idx(i,p,n)], aq = M.d[idx(i,q,n)];
        double apn =  c*ap - s*aq;
        double aqn =  s*ap + c*aq;
        M.d[idx(i,p,n)] = apn;
        M.d[idx(i,q,n)] = aqn;
    }
}

// swap cols
void swap_cols(MATRIX M, int a, int b){
    int m=M.rows, n=M.cols;
    for (int i=0;i<m;i++){
        double tmp = M.d[idx(i,a,n)];
        M.d[idx(i,a,n)] = M.d[idx(i,b,n)];
        M.d[idx(i,b,n)] = tmp;
    }
}

// finding theta
int jacobi_angle_cs(MATRIX A, int p, int q, double *c, double *s){
    double alpha = col_dot(A,p,p);
    double beta  = col_dot(A,q,q);
    double gamma = col_dot(A,p,q);
    double tol2  = TOL*TOL;
    if (gamma*gamma <= tol2 * alpha * beta) return 0;  // orthogonality
    double theta = 0.5 * atan2(2.0*gamma, beta - alpha);
    *c = cos(theta); *s = sin(theta);
    return 1;
}

// One sided Jacobi
int jacobi_topK(MATRIX Ain, int k, MATRIX *A_copy, MATRIX *Vt)
{
    int m = Ain.rows;
    int n = Ain.cols;

    // copy of A 
    *A_copy = mat_alloc(m, n);
    if (!A_copy->d) return -1;
    for (int t = 0; t < m * n; ++t) {
        A_copy->d[t] = Ain.d[t];
    }

   // V=I (initially)
    *Vt = mat_alloc(n, n);

    mat_identity(*Vt);

// Jacobi Sweep
    for (int sweep = 0; sweep < MAX_SWEEPS; sweep++) {
        int flag = 0;

        // every (p, q) with p < q
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double c, s;

                if (jacobi_angle_cs(*A_copy, p, q, &c, &s)==0) {
                    continue; // orthogonality achieved
                }
                // alter V
                rotate_cols(*A_copy,     p, q, c, s);
                rotate_cols(*Vt, p, q, c, s);
                flag = 1;
            }
        }
        // converged
        if (flag==0) break;
    }

// Compt eigen value
    double *sigma_val = (double*)malloc((size_t)n * sizeof(double));

    
    for (int j = 0; j < n; j++) {
        sigma_val[j] = col_norm(*A_copy, j);
    }

    // align eigen vector
    for (int t = 0; t < k; t++) {
        int best = t;
        for (int j = t + 1; j < n; j++) {
            if (sigma_val[j] > sigma_val[best]) best = j;
        }
        if (best != t) {
            double tmp = sigma_val[best]; sigma_val[best] = sigma_val[t]; sigma_val[t] = tmp;
            swap_cols(*A_copy,     best, t);
            swap_cols(*Vt, best, t);
        }
    }

    free(sigma_val);
    return 0;
}

// Finding SVD-Truncated
void reconstruct_direct(MATRIX Ak, MATRIX A_copy, MATRIX Vfull, int k){
    int m=Ak.rows, n=Ak.cols;
    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            double sum=0.0;
            for (int t=0;t<k;t++){
                sum += A_copy.d[idx(i,t,n)] * Vfull.d[idx(j,t,n)];
            }
            Ak.d[idx(i,j,n)] = sum;
        }
    }
}



int main(){
    char inpath[512], outpath[512];
int k;

// Input Path
printf("Input image path (.jpg/.jpeg/.png): ");
if (!fgets(inpath, sizeof inpath, stdin)) return 0;
for (char *p = inpath; *p; ++p) {
    if (*p == '\n' || *p == '\r') { *p = '\0'; break; }
}
// Output Path
printf("Output PGM path (.pgm): ");
if (!fgets(outpath, sizeof outpath, stdin)) return 0;
for (char *p = outpath; *p; ++p) {
    if (*p == '\n' || *p == '\r') { *p = '\0'; break; }
}

printf("Rank k (integer): ");
if (scanf("%d", &k) != 1) return 0;

// Things to be done
//  Load
//  Jacobi
//  Reconstruct
//  Save
//  Free


//  Load
    MATRIX A;
    if (stb_read_gray_matrix(inpath, &A)!=0){
        fprintf(stderr,"Image load failed\n");
        return 0;
    }

    //  Jacobi
    MATRIX A_copy, Vfull;
    if (jacobi_topK(A, k, &A_copy, &Vfull)!=0){
        mat_free(&A);
        return 0;
    }
    //  Reconstruct
    MATRIX Ak = mat_alloc(A.rows, A.cols);
    if (!Ak.d){ mat_free(&A_copy); mat_free(&Vfull); mat_free(&A); return 0; }
    reconstruct_direct(Ak, A_copy, Vfull, k);

    if (pgm_write(outpath, Ak)!=0){
        fprintf(stderr,"Failed to write PGM\n");
    }
    // Free
    mat_free(&A_copy);
    mat_free(&Vfull);
    mat_free(&Ak);
    mat_free(&A);
    return 0;
}