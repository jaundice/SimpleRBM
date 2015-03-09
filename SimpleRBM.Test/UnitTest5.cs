using System;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Demo.Util;

namespace SimpleRBM.Test
{
    [TestClass]
    public class GrayCodes
    {


        [TestMethod]
        public void TestULongs()
        {
            var vals = Enumerable.Range(50000, 100000).Select(a =>
            {
                var orig = (ulong)a;
                var coded = (GrayCodeU64)orig;
                var decoded = (ulong)coded;
                Assert.AreEqual(orig, decoded);
                return string.Format("{0}\t{1}\t{2}", a, GrayCodeU64.DebugString(coded), decoded);
            }).ToList();
            vals.ForEach(Console.WriteLine);
        }

        [TestMethod]
        public void TestUInts()
        {
            var vals = Enumerable.Range(0, 5000).Select(a =>
            {
                var orig = (uint)a;
                var coded = (GrayCodeU32)orig;
                var decoded = (uint)coded;
                Assert.AreEqual(orig, decoded);
                return string.Format("{0}\t{1}\t{2}\t[{3}]", a, GrayCodeU32.DebugString(coded), decoded, string.Join(",", GrayCodeU32.GetSetBits(coded, '1', '0')));
            }).ToList();
            vals.ForEach(Console.WriteLine);
        }

        [TestMethod]
        public void TestUShort()
        {
            var vals = Enumerable.Range(0, ushort.MaxValue).Select(a =>
            {
                var orig = (ushort)a;
                var coded = (GrayCodeU16)orig;
                var decoded = (ushort)coded;
                Assert.AreEqual(orig, decoded);
                return string.Format("{0}\t{1}\t{2}\t[{3}]", a, GrayCodeU16.DebugString(coded), decoded, string.Join(",", GrayCodeU16.GetSetBits(coded, '1', '0')));
            }).ToList();
            vals.ForEach(Console.WriteLine);
        }

        [TestMethod]
        public void TestByte()
        {
            var vals = Enumerable.Range(0, 256).Select(a =>
            {
                var orig = (byte)a;
                var coded = (GrayCodeU8)orig;
                var decoded = (byte)coded;
                Assert.AreEqual(orig, decoded);
                return string.Format("{0}\t{1}\t{2}\t[{3}]", a, GrayCodeU8.DebugString(coded), decoded, string.Join(",", GrayCodeU8.GetSetBits(coded, '1', '0')));
            }).ToList();
            vals.ForEach(Console.WriteLine);
        }
    }


}