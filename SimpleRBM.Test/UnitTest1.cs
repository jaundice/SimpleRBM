using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Common.ExitCondition;

namespace SimpleRBM.Test
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            var buffer = new CircularBuffer<double>(20);

            for (int i = 0; i < 1001; i++)
            {
                buffer.Add(i);
            }

            Console.WriteLine("Min Seen:{0} Max Seen:{1}", buffer.MinValueSeen, buffer.MaxValueSeen);

            foreach (var d in buffer)
            {
                Console.WriteLine(d);
            }
        }
    }
}
