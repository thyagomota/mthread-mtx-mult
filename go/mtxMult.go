/*
 * Project Name: mthread-mtx-mult
 * Project Description: Multithreaded Matrix Multiplication Performance Evaluation
 * Author: Thyago Mota (MSU Denver)
 * Contributors:
 * Date: 2020-12-01
 */

package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// type definitions
type matrix [][]int

// globals
var wg sync.WaitGroup
const MAX_INT = 10
const DISPLAY_MATRICES = true

/*
 * generates a random nxn matrix of integers
 */
func randomMatrix(n int) matrix {
	mtx := make(matrix, n)
	for i := range mtx {
		mtx[i] = make([]int, n)
		for j := 0; j < n; j++ {
			mtx[i][j] = rand.Intn(MAX_INT)
		}
	}
	return mtx
}

/*
 * generates a all-ones nxn matrix of integers
 */
func allOnesMatrix(n int) matrix {
	mtx := make(matrix, n)
	for i := range mtx {
		mtx[i] = make([]int, n)
		for j := 0; j < n; j++ {
			mtx[i][j] = 1
		}
	}
	return mtx
}

/*
 * generates a nxn matrix of integers based on a given string; the size of the matrix is determined by the number of lines of the string
 */
func stringMatrix(s string) matrix {
	n := len(strings.Split(s, "\n"))
	mtx := make(matrix, n)
	sc := bufio.NewScanner(strings.NewReader(s))
	i := 0
	for sc.Scan() {
		mtx[i] = make([]int, n)
		line := sc.Text()
		for j, el := range strings.Split(line, " ") {
			x, _ := strconv.Atoi(el)
			mtx[i][j] = x
		}
		i++
	}
	return mtx
}

/*
 * returns a string representation of a given matrix; parameter col defines the size of each col for better visualization
 */
func (mtx matrix) toString(col int) string {
	s := ""
	for i := 0; i < len(mtx); i++ {
		for j := 0; j < len(mtx[0]); j++ {
			s += fmt.Sprintf("%" + strconv.FormatInt(int64(col), 10) + "d ", mtx[i][j])
		}
		s += "\n"
	}
	s = strings.Trim(s, "\n")
	return s
}

/*
 * returns the sub-matrix (i, j) by slicing the callee matrix using slice parameter "s"
 */
func (mtx matrix) getSlice(i int, j int, s int) matrix {
	slc := make(matrix, s)
	row := i * s
	for newMtxRow := 0; newMtxRow < s; newMtxRow++ {
		slc[newMtxRow] = make([]int, s)
		col := j * s
		for newMtxCol := 0; newMtxCol < s; newMtxCol++{
			slc[newMtxRow][newMtxCol] = mtx[row][col]
			col++
		}
		row++
	}
	return slc
}

/*
 * returns ALL slices of the callee matrix using slice parameter "s"
 */
func (mtx matrix) getSlices(s int) [][]matrix {
	n := len(mtx)
	slices := make([][]matrix, n / s)
	for i := 0; i < n / s; i++ {
		slices[i] = make([]matrix, n / s)
		for j := 0; j < n / s; j++ {
			slices[i][j] = mtx.getSlice(i, j, s)
		}
	}
	return slices
}

/*
 * merges ALL given slices into the callee matrix
 */
func (mtx matrix) mergeSlices(slices [][]matrix) {
	n := len(mtx)
	s := len(slices[0][0])
	row := 0
	for i := 0; i < n; i++ {
		col := 0
		for j := 0; j < n; j++ {
			mtx[i][j] = slices[i / s][j / s][row][col]
			col++
			col = col % s
		}
		row++
		row = row % s
	}
}

/*
 * performs single-threaded "st" matrix multiplication
 */
func stMultiply(mtxA [][]int, mtxB [][]int) matrix {
	n := len(mtxA)
	newMtx := make(matrix, n)
	for i := 0; i < n; i++ {
		newMtx[i] = make([]int, n)
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				newMtx[i][j] += mtxA[i][k] * mtxB[k][j];
			}
		}
	}
	return newMtx
}

/*
 * multiplies the two given matrices, adding the result into the callee matrix
 */
func (mtxC matrix) addMultiply(mtxA matrix, mtxB matrix)  {
	defer wg.Done()
	n := len(mtxA)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				mtxC[i][j] += mtxA[i][k] * mtxB[k][j]
			}
		}
	}
}

/*
 * performs multi-threaded "mt" matrix multiplication
 */
func mtMultiply(mtxA matrix, mtxB matrix, s int) matrix {
	// step 1: slice each matrix
	slicesA := mtxA.getSlices(s)
	slicesB := mtxB.getSlices(s)

	// step 2: allocate the resulting matrix
	n := len(mtxA)
	mtxC := make(matrix, n)
	for i := 0; i < n; i++ {
		mtxC[i] = make([]int, n)
	}

	// step 3: slice the resulting matrix
	slicesC := mtxC.getSlices(s)

	// step 4: multiply the slices in parallel
	for i := 0; i < n / s; i++ {
		for j := 0; j < n / s; j++ {
			wg.Add(n / s)
			for k := 0; k < n / s; k++ {
				go slicesC[i][j].addMultiply(slicesA[i][k], slicesB[k][j])
			}
			wg.Wait()
		}
	}

	// step 5: merge the slices to reconstruct the resulting matrix
	mtxC.mergeSlices(slicesC)
	return mtxC
}

func usage() {
	println("Use: go run mtxMult n s")
	println("\tn: size of the matrices (n >= 4)")
	println("\ts: size of each slice (n % s = 0)")
}

/*
 * generate two random nxn matrices and time the single and multi-threaded multiplication; accepted command-line parameters are: "n" (size of the matrices) and "s" (size of each matrix slice)
 */
func main() {
	// command-line validation
	args := os.Args[1:]
	if (len(args) != 2) {
		usage()
		os.Exit(1)
	}
	n, errN := strconv.Atoi(args[0])
	s, errS := strconv.Atoi(args[1])
	if errN != nil || errS != nil || n < 4 || n % s != 0 {
		usage()
		os.Exit(1)
	}
	println("Parameters: n=", n, "; s=", s)

	// generating the matrices
	// mtxA := randomMatrix(n)
	mtxA := allOnesMatrix(n)
	if DISPLAY_MATRICES {
		println("Matrix A")
		println(mtxA.toString(4))
		println()
	}
	// mtxB := randomMatrix(n)
	mtxB := allOnesMatrix(n)
	if DISPLAY_MATRICES {
		println("Matrix B")
		println(mtxB.toString(4))
		println()
	}

	// timing single-threaded multiplication
	println("Single-threaded multiplication...")
	start := time.Now()
	mtxC := stMultiply(mtxA, mtxB)
	elapsed := time.Since(start)
	fmt.Printf("Done! It took %vms\n", elapsed.Milliseconds())
	if DISPLAY_MATRICES {
		println("Matrix C")
		println(mtxC.toString(4))
		println()
	}

	// timing multi-threaded multiplication
	println("Multi-threaded multiplication...")
	start = time.Now()
	mtxC = mtMultiply(mtxA, mtxB, s)
	elapsed = time.Since(start)
	fmt.Printf("Done! It took %vms\n", elapsed.Milliseconds())
	if DISPLAY_MATRICES {
		println("Matrix C")
		println(mtxC.toString(4))
		println()
	}
}
