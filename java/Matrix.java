/*
 * Project Name: mthread-mtx-mult
 * Project Description: Multithreaded Matrix Multiplication Performance Evaluation
 * Author: Thyago Mota (MSU Denver)
 * Contributors:
 * Date: 2020-12-02
 */

import java.util.Random;
import java.util.concurrent.CountDownLatch;

public class Matrix {

    private int data[][];
    private static final int MAX_INT = 10;
    private static final int FORMAT_COLS = 4;

    public Matrix(int n) {
        data = new int[n][n];
    }

    public Matrix(String s) {
        String lines[] = s.split("\n");
        int n = lines.length;
        data = new int[n][n];
        for (int i = 0; i < data.length; i++) {
            String line = lines[i];
            String cols[] = line.split(" ");
            for (int j = 0; j < cols.length; j++)
                data[i][j] = Integer.parseInt(cols[j]);
        }
    }

    public void setAllZeros() {
        for (int i = 0; i < data.length; i++)
            for (int j = 0; j < data[0].length; j++)
                data[i][j] = 0;
    }

    public void setAllOnes() {
        for (int i = 0; i < data.length; i++)
            for (int j = 0; j < data[0].length; j++)
                data[i][j] = 1;
    }

    public void setRandom() {
        Random r = new Random();
        for (int i = 0; i < data.length; i++)
            for (int j = 0; j < data[0].length; j++)
                data[i][j] = r.nextInt(MAX_INT);
    }

    @Override
    public String toString() {
        String s = "";
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++)
                s += String.format("%" + FORMAT_COLS + "d ", data[i][j]);
            s += "\n";
        }
        s = s.substring(0, s.length() - 1);
        return s;
    }

    /*
     * returns the sub-matrix (i, j) by slicing the callee matrix using slice parameter "s"
     */
    public Matrix getSlice(int i , int j, int s) {
        Matrix slc = new Matrix(s);
        int row = i * s;
        for (int slcRow = 0; slcRow < s; slcRow++) {
            int col = j * s;
            for (int slcCol = 0; slcCol < s; slcCol++) {
                slc.data[slcRow][slcCol] = data[row][col];
                col++;
            }
            row++;
        }
        return slc;
    }

    /*
     * returns ALL slices of the callee matrix using slice parameter "s"
     */
    public Matrix[][] getSlices(int s) {
        int n = data.length;
        Matrix slices[][] = new Matrix[n / s][n / s];
        for (int i = 0; i < n / s; i++)
            for (int j = 0; j < n / s; j++)
                slices[i][j] = getSlice(i, j, s);
        return slices;
    }

    /*
     * merges ALL given slices into the callee matrix
     */
    public void mergeSlices(Matrix[][] slices) {
        int n = data.length;
        int s = slices[0][0].data.length;
        int row = 0;
        for (int i = 0; i < n; i++) {
            int col = 0;
            for (int j = 0; j < n; j++) {
                data[i][j] = slices[i / s][j / s].data[row][col];
                col++;
                col = col % s;
            }
            row++;
            row = row % s;
        }
    }

    /*
     * performs single-threaded "st" matrix multiplication
     */
    public static Matrix stMultiply(Matrix mtxA, Matrix mtxB) {
        int n = mtxA.data.length;
        Matrix newMtx = new Matrix(n);
        newMtx.setAllZeros();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    newMtx.data[i][j] += mtxA.data[i][k] * mtxB.data[k][j];
        return newMtx;
    }

    /*
     * multiplies the two given matrices, adding the result into the callee matrix
     */
    public void addMultiply(Matrix mtxA, Matrix mtxB) {
        int n = mtxA.data.length;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    data[i][j] += mtxA.data[i][k] * mtxB.data[k][j];
    }

    /*
     * performs multi-threaded "mt" matrix multiplication
     */
    public static Matrix mtMultiply(Matrix mtxA, Matrix mtxB, int s) throws InterruptedException {
        // step 1: slice each matrix
        Matrix slicesA[][] = mtxA.getSlices(s);
        Matrix slicesB[][] = mtxB.getSlices(s);

        // step 2: allocate the resulting matrix
        int n = mtxA.data.length;
        Matrix mtxC = new Matrix(n);
        mtxC.setAllZeros();

        // step 3: slice the resulting matrix
        Matrix slicesC[][] = mtxC.getSlices(s);

        // step 4: multiply the slices in parallel
        for (int i = 0; i < n / s; i++)
            for (int j = 0; j < n / s; j++) {
                CountDownLatch latch = new CountDownLatch(n / s);
                for (int k = 0; k < n / s; k++) {
                    MultThread multThread = new MultThread(latch, slicesA[i][k], slicesB[k][j], slicesC[i][j]);
                    new Thread(multThread).start();
                }
                latch.await();
            }

        // step 5: merge the slices to reconstruct the resulting matrix
        mtxC.mergeSlices(slicesC);
        return mtxC;
    }
}
