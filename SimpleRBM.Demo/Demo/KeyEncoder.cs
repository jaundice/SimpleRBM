using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SimpleRBM.Demo.Demo
{
    public class KeyEncoder
    {
        public static ulong[] GenerateKeys<T>(T[,] encoded) where T : struct, IComparable<T>
        {
            var keys = new ulong[encoded.GetLength(0)];
            int width = encoded.GetLength(1);
            Parallel.For(0, keys.Length, a =>
            {
                ulong v = 0;
                for (int i = 0; i < width; i++)
                {
                    if (Comparer<T>.Default.Compare(encoded[a, i], default(T)) > 0)
                    {
                        v |= ((ulong) 1u << (width - i));
                    }
                }
                keys[a] = v;
            });
            return keys;
        }
    }
}