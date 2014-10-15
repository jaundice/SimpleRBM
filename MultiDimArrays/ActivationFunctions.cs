﻿using System;
using System.Threading.Tasks;

namespace MultidimRBM
{
    public static class ActivationFunctions
    {
        /// <summary>
        ///     Basic sigmoid function
        /// </summary>
        /// <param name="x">Real number</param>
        /// <returns>Real number</returns>
        public static double Logistic(double x)
        {
            return 1d/(1d + Math.Exp(-x));
        }

        /// <summary>
        ///     Apply a logistic function on all elements of a matrix
        /// </summary>
        /// <param name="matrix">Matrix</param>
        /// <returns>Logistic matrix</returns>
        public static double[,] Logistic(double[,] matrix)
        {
            var result = new double[matrix.GetLength(0), matrix.GetLength(1)];

            Parallel.For(0, matrix.GetLength(0),
                i => Parallel.For(0, matrix.GetLength(1), j => { result[i, j] = Logistic(matrix[i, j]); }));
            return result;
        }

        //public static RVector Logistic(RVector vector)
        //{
        //    var result = new RVector(vector.Length);

        //    for (int i = 0; i < vector.Length; i++)
        //    {
        //        result[i] = Logistic(vector[i]);
        //    }
        //    return result;
        //}
    }
}