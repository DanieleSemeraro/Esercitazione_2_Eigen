#include <iostream>
#include "Eigen/Eigen"

using namespace Eigen;
using namespace std;

Vector2d PALU(const Matrix2d& A, const Vector2d& b) {//Risoluzione PALU con pivoting parziale
    PartialPivLU<Matrix2d> lu(A);                   // ovvero scambio di sole righe
    Vector2d x = lu.solve(b);                      //(dovrebbe essere più veloce)
    return x;
}
Vector2d PALU2(const Matrix2d& A, const Vector2d& b) {//Risoluzione PALU con Pivoting totale
    FullPivLU<Matrix2d> lu(A);                       // ovvero scambio di righe e colonne
    Vector2d x = lu.solve(b);
    return x;
}

Vector2d QR(const Matrix2d& A, const Vector2d& b) {//Risoluzione QR con sola riflessione di Householder
    HouseholderQR<Matrix2d> qr(A);
    Vector2d x = qr.solve(b);
    return x;
}
Vector2d QR2(const Matrix2d& A, const Vector2d& b) {//Risoluzione QR con Householder e Pivoting colonne
    ColPivHouseholderQR<Matrix2d> qr(A);
    Vector2d x = qr.solve(b);
    return x;
}

double errore_relativo(const Vector2d& x, const Vector2d& x_atteso) {
    return ((x - x_atteso).norm() / x_atteso.norm())*100;
}//per l'errore relativo divido il valore assoluto per il valore atteso e moltiplico per 100 per averlo in percentuale

int main() {
    Matrix2d A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    Matrix2d A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    Matrix2d A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    Vector2d x_atteso(2);//soluzione reale
    x_atteso << -1, -1;
    //SOLUZIONI SISTEMI
    Vector2d x1_PALU = PALU(A1, b1);
    Vector2d x2_PALU = PALU(A2, b2);
    Vector2d x3_PALU = PALU(A3, b3);
    Vector2d x1_QR = QR(A1, b1);
    Vector2d x2_QR = QR(A2, b2);
    Vector2d x3_QR = QR(A3, b3);
    //CALCOLO ERRORE RELATIVO PER OGNI RISOLUZIONE
    double error1_PALU = errore_relativo(x1_PALU,x_atteso);
    double error2_PALU = errore_relativo(x2_PALU,x_atteso);
    double error3_PALU = errore_relativo(x3_PALU,x_atteso);
    double error1_QR = errore_relativo(x1_QR,x_atteso);
    double error2_QR = errore_relativo(x2_QR,x_atteso);
    double error3_QR = errore_relativo(x3_QR,x_atteso);
    cout << "ERRORE Soluzione PALU:" << endl;
    cout << "System 1 Relative Error: " << error1_PALU <<"%"<< endl;
    cout << "System 2 Relative Error: " << error2_PALU <<"%"<< endl;
    cout << "System 3 Relative Error: " << error3_PALU <<"%"<< endl;

    cout << "ERRORE Soluzione QR:" << endl;
    cout << "System 1 Relative Error: " << error1_QR <<"%"<< endl;
    cout << "System 2 Relative Error: " << error2_QR <<"%"<< endl;
    cout << "System 3 Relative Error: " << error3_QR <<"%"<< endl;
    //RIPETO CON PIVOTING TOTALE E PIVOTING+QR
    Vector2d x1_PALU2 = PALU2(A1, b1);
    Vector2d x2_PALU2 = PALU2(A2, b2);
    Vector2d x3_PALU2 = PALU2(A3, b3);
    Vector2d x1_QR2 = QR2(A1, b1);
    Vector2d x2_QR2 = QR2(A2, b2);
    Vector2d x3_QR2 = QR2(A3, b3);
    double error1_PALU2 = errore_relativo(x1_PALU2,x_atteso);
    double error2_PALU2 = errore_relativo(x2_PALU2,x_atteso);
    double error3_PALU2= errore_relativo(x3_PALU2,x_atteso);
    double error1_QR2 = errore_relativo(x1_QR2,x_atteso);
    double error2_QR2 = errore_relativo(x2_QR2,x_atteso);
    double error3_QR2 = errore_relativo(x3_QR2,x_atteso);
    cout << "ERRORE Soluzione PALU2:" << endl;
    cout << "System 1 Relative Error: " << error1_PALU2 <<"%"<< endl;
    cout << "System 2 Relative Error: " << error2_PALU2 <<"%"<< endl;
    cout << "System 3 Relative Error: " << error3_PALU2 <<"%"<< endl;

    cout <<"ERRORE Soluzione QR2:" << endl;
    cout << "System 1 Relative Error: " << error1_QR2 <<"%"<< endl;
    cout << "System 2 Relative Error: " << error2_QR2 <<"%"<< endl;
    cout << "System 3 Relative Error: " << error3_QR2 <<"%"<< endl;

    return 0;
}
