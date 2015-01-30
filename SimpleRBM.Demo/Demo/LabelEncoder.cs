using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleRBM.Demo.Demo
{
    public static class LabelEncoder
    {
        public static T[,] EncodeLabels<TLabel, T>(TLabel[] labels, int numDistinctLabelOptions = 0)
        {
            List<TLabel> distinct = labels.Distinct().OrderBy(a => a).ToList();
            Dictionary<TLabel, int> indices = distinct.ToDictionary(a => a, distinct.IndexOf);

            var on = (T) Convert.ChangeType(1.0, typeof (T));

            if (numDistinctLabelOptions == 0)
                numDistinctLabelOptions = distinct.Count;

            var array = new T[labels.Length, numDistinctLabelOptions];
            for (int i = 0; i < labels.Length; i++)
            {
                array[i, indices[labels[i]]] = on;
            }
            return array;
        }

        public static T[,] EncodeLabels<TLabel, T>(TLabel[] labels, TLabel[] distinctLabelOptions)
        {
            List<TLabel> distinct = distinctLabelOptions.OrderBy(a => a).ToList();
            Dictionary<TLabel, int> indices = distinct.ToDictionary(a => a, distinct.IndexOf);

            var on = (T)Convert.ChangeType(1.0, typeof(T));


            var array = new T[labels.Length, distinctLabelOptions.Length];
            for (int i = 0; i < labels.Length; i++)
            {
                array[i, indices[labels[i]]] = on;
            }
            return array;
        }
    }
}