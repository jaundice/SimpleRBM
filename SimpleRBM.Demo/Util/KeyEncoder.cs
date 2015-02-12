using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace SimpleRBM.Demo.Util
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

        public static string[] CreateBinaryStringKeys<T>(T[,] encoded)
        {
            if (encoded == null)
                return null;

            int width = encoded.GetLength(1);
            var ret = new string[encoded.GetLength(0)];
            for (var a = 0; a < encoded.GetLength(0); a++)
            {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < width; i++)
                {


                    if (Comparer<T>.Default.Compare(encoded[a, i], default(T)) > 0)
                        sb.Append("1");
                    else
                        sb.Append("0");
                }
                ret[a] = sb.ToString();


            }
            return ret;
        }

    }


}