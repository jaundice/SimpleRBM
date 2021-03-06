﻿using System;
using System.Linq;
using System.Text.RegularExpressions;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    public class CommandLine
    {
        public static bool ParseLayerDefinition(string s, out ILayerDefinition ret)
        {
            if (Regex.IsMatch(s, @"\d+x\d+"))
            {

                string[] parts = s.Split(new[] { 'x', 'X' }, StringSplitOptions.RemoveEmptyEntries);
                ret = new LayerDefinition()
                {
                    VisibleUnits = int.Parse(parts[0]),
                    HiddenUnits = int.Parse(parts[1])
                };

                return true;
            }
            ret = null;
            return false;
        }

        public static bool ParseLayerDefinitionArray(string s, out ILayerDefinition[] ret)
        {
            return ParseArray(s, ParseLayerDefinition, out ret);
        }

        public static T ReadCommandLine<T>(string prefix, Parse<T> parser, T defaultVal)
        {
            string s =
                Environment.GetCommandLineArgs()
                    .FirstOrDefault(a => a.StartsWith(prefix, StringComparison.CurrentCultureIgnoreCase));

            if (s == null)
                return defaultVal;
            T val;
            return parser(s.Substring(s.IndexOf(":") + 1), out val) ? val : defaultVal;
        }

        public static bool FakeParseString(string s, out string v)
        {
            v = s;
            return s != null;
        }

        private static bool ParseArray<T>(string s, Parse<T> inner, out T[] ret)
        {
            T[] arr = string.IsNullOrEmpty(s)
                ? new T[0]
                : s
                    .Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(a =>
                    {
                        T temp;
                        bool c = inner(a, out temp);
                        return new
                        {
                            success = c,
                            value = temp
                        };
                    }).Where(a => a.success).Select(a => a.value)
                    .ToArray();
            ret = arr;

            return !string.IsNullOrEmpty(s);
        }

        public static bool ParseIntArray(string s, out int[] ret)
        {
            return ParseArray(s, int.TryParse, out ret);
        }

        public delegate bool Parse<T>(string s, out T res);
    }
}