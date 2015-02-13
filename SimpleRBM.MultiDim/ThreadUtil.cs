using System;
using System.Threading.Tasks;

namespace SimpleRBM.MultiDim
{
    public static class ThreadUtil
    {
        private const int MAX = 1024;

        public static void Run<TElement>(TElement[,] arr, Action<int, int> action)
        {
            Run(arr.GetLength(0), arr.GetLength(1), action);
        }

        public static void Run(int rows, int cols, Action<int, int> action)
        {
            if (rows*cols < MAX)
            {
                Parallel.For(0, rows, i => Parallel.For(0, cols, j => action(i, j)));
            }
            else if (cols > rows)
            {
                for (var i = 0; i < rows; i++)
                {
                    var ii = i;
                    Parallel.For(0, cols, j => action(ii, j));
                }
            }
            else
            {
                Parallel.For(0, rows, i =>
                {
                    for (var j = 0; j < cols; j++)
                    {
                        action(i, j);
                    }
                });
            }
        }
    }
}