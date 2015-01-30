using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

namespace SimpleRBM.Demo.Demo
{
    public class KeyEncoder
    {
        public static ulong[][] GenerateKeys<T>(T[,] encoded) where T : struct, IComparable<T>
        {
            if (encoded == null)
                return null;

            int width = encoded.GetLength(1);

            var keyElementWidth = (int)Math.Ceiling((double)width / 64);
            var keys = new ulong[encoded.GetLength(0)][];


            Parallel.For(0, keys.GetLength(0), a =>
            {
                ulong[] key = new ulong[keyElementWidth];


                var rowIdx = 0;
                for (int i = 0; i < width; i++)
                {
                    var ndx = i % 64;

                    if (ndx == 0 && i > 0)
                        rowIdx++;

                    if (Comparer<T>.Default.Compare(encoded[a, i], default(T)) > 0)
                    {
                        key[rowIdx] |= ((ulong)1u << ndx); //(64 - ndx));
                    }
                }
                keys[a] = key;
            });
            return keys;
        }
    }


}