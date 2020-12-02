/*
 * Project Name: mthread-mtx-mult
 * Project Description: Multithreaded Matrix Multiplication Performance Evaluation
 * Author: Thyago Mota (MSU Denver)
 * Contributors:
 * Date: 2020-12-02
 */

public class MtxMult {

    public static final boolean DISPLAY_MATRICES = false;

    public static void usage() {
        System.out.println("Use: java MtxMult n s");
        System.out.println("\tn: size of the matrices (n >= 4)");
        System.out.println("\ts: size of each slice (n % s = 0)");
    }

    public static void main(String[] args) throws InterruptedException {
        // command-line validation
        if (args.length != 2) {
            usage();
            System.exit(1);
        }
        int n = 0;
        int s = 0;
        try {
            n = Integer.parseInt(args[0]);
            s = Integer.parseInt(args[1]);
        }
        catch (NumberFormatException ex) {
            usage();
            System.exit(1);
        }
        if (n < 4 || n % s != 0) {
            usage();
            System.exit(1);
        }
        System.out.println("Parameters: n=" + n + "; s=" + s);

        // generating the matrices
        Matrix mtxA = new Matrix(n);
        mtxA.setAllOnes();
        if (DISPLAY_MATRICES) {
            System.out.println("Matrix A");
            System.out.println(mtxA);
            System.out.println();
        }
        Matrix mtxB = new Matrix(n);
        mtxB.setAllOnes();
        if (DISPLAY_MATRICES) {
            System.out.println("Matrix B");
            System.out.println(mtxB);
            System.out.println();
        }

        // timing single-threaded multiplication
        System.out.println("Single-threaded multiplication...");
        long start = System.nanoTime();
        Matrix mtxC = Matrix.stMultiply(mtxA, mtxB);
        long elapsed = (System.nanoTime() - start) / 1000000;
        System.out.println("Done! It took " + elapsed + "ms");
        if (DISPLAY_MATRICES) {
            System.out.println("Matrix C");
            System.out.println(mtxC);
            System.out.println();
        }

        // timing multi-threaded multiplication
        System.out.println("Multi-threaded multiplication...");
        start = System.nanoTime();
        mtxC = Matrix.mtMultiply(mtxA, mtxB, s);
        elapsed = (System.nanoTime() - start) / 1000000;
        System.out.println("Done! It took " + elapsed + "ms");
        if (DISPLAY_MATRICES) {
            System.out.println("Matrix C");
            System.out.println(mtxC);
            System.out.println();
        }
    }
}
