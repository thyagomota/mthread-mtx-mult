/*
 * Project Name: mthread-mtx-mult
 * Project Description: Multithreaded Matrix Multiplication Performance Evaluation
 * Author: Thyago Mota (MSU Denver)
 * Contributors:
 * Date: 2020-12-02
 */

import java.util.concurrent.CountDownLatch;

public class MultThread implements Runnable {

    private CountDownLatch latch;
    private Matrix mtxA, mtxB, mtxC;

    public MultThread(CountDownLatch latch, final Matrix mtxA, final Matrix mtxB, Matrix mtxC) {
        this.latch = latch;
        this.mtxA = mtxA;
        this.mtxB = mtxB;
        this.mtxC = mtxC;
    }

    @Override
    public void run() {
        mtxC.addMultiply(mtxA, mtxB);
        latch.countDown();
    }
}
