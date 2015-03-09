using System;
using System.Text;

namespace SimpleRBM.Demo.Util
{
    public struct GrayCodeU8
    {
        private GrayCodeU8(byte binary)
            : this()
        {
            Code = (byte) ((binary >> 1) ^ binary);
        }

        public byte Code { get; private set; }

        public static string DebugString(GrayCodeU8 code)
        {
            return new string(GetSetBits(code, '1', '0'));
        }


        public static T[] GetSetBits<T>(GrayCodeU8 code, T trueValue, T falseValue)
        {
            var ret = new T[8];
            SetBits(code, ret, 0, trueValue, falseValue);
            return ret;
        }

        public static void SetBits<T>(GrayCodeU8 code, T[] target, int targetOffset, T trueValue, T falseValue)
        {
            target[targetOffset + 0] = (code.Code & 0x1b << 7) == 0 ? falseValue : trueValue;
            target[targetOffset + 1] = (code.Code & 0x1b << 6) == 0 ? falseValue : trueValue;
            target[targetOffset + 2] = (code.Code & 0x1b << 5) == 0 ? falseValue : trueValue;
            target[targetOffset + 3] = (code.Code & 0x1b << 4) == 0 ? falseValue : trueValue;
            target[targetOffset + 4] = (code.Code & 0x1b << 3) == 0 ? falseValue : trueValue;
            target[targetOffset + 5] = (code.Code & 0x1b << 2) == 0 ? falseValue : trueValue;
            target[targetOffset + 6] = (code.Code & 0x1b << 1) == 0 ? falseValue : trueValue;
            target[targetOffset + 7] = (code.Code & 0x1b) == 0 ? falseValue : trueValue;
        }

        public static void SetBits<T>(GrayCodeU8 code, T[,] target, int targetRow, int targetOffset, T trueValue,
            T falseValue)
        {
            target[targetRow, targetOffset + 0] = (code.Code & 0x1b << 7) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 1] = (code.Code & 0x1b << 6) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 2] = (code.Code & 0x1b << 5) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 3] = (code.Code & 0x1b << 4) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 4] = (code.Code & 0x1b << 3) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 5] = (code.Code & 0x1b << 2) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 6] = (code.Code & 0x1b << 1) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 7] = (code.Code & 0x1b) == 0 ? falseValue : trueValue;
        }

        public static GrayCodeU8 Encode(byte binary)
        {
            return new GrayCodeU8(binary);
        }

        public static byte Decode(GrayCodeU8 code)
        {
            byte mask;
            byte num = code.Code;
            for (mask = (byte) (num >> 1); mask != 0; mask = (byte) (mask >> 1))
            {
                num = (byte) (num ^ mask);
            }
            return num;
        }

        public static byte Decode(byte code)
        {
            var c = new GrayCodeU8 {Code = code};
            return Decode(c);
        }


        public static explicit operator GrayCodeU8(byte num)
        {
            return new GrayCodeU8(num);
        }

        public static explicit operator byte(GrayCodeU8 code)
        {
            return Decode(code);
        }
    }

    public struct GrayCodeU16
    {
        private GrayCodeU16(ushort binary)
            : this()
        {
            Code = (ushort) ((binary >> 1) ^ binary);
        }

        public ushort Code { get; private set; }

        public static string DebugString(GrayCodeU16 code)
        {
            return new string(GetSetBits(code, '1', '0'));
        }

        public static void SetBits<T>(GrayCodeU16 code, T[] target, int targetOffset, T trueValue, T falseValue)
        {
            target[targetOffset + 0] = (code.Code & (ushort) 0x1 << 15) == 0 ? falseValue : trueValue;
            target[targetOffset + 1] = (code.Code & (ushort) 0x1 << 14) == 0 ? falseValue : trueValue;
            target[targetOffset + 2] = (code.Code & (ushort) 0x1 << 13) == 0 ? falseValue : trueValue;
            target[targetOffset + 3] = (code.Code & (ushort) 0x1 << 12) == 0 ? falseValue : trueValue;
            target[targetOffset + 4] = (code.Code & (ushort) 0x1 << 11) == 0 ? falseValue : trueValue;
            target[targetOffset + 5] = (code.Code & (ushort) 0x1 << 10) == 0 ? falseValue : trueValue;
            target[targetOffset + 6] = (code.Code & (ushort) 0x1 << 9) == 0 ? falseValue : trueValue;
            target[targetOffset + 7] = (code.Code & (ushort) 0x1 << 8) == 0 ? falseValue : trueValue;
            target[targetOffset + 8] = (code.Code & (ushort) 0x1 << 7) == 0 ? falseValue : trueValue;
            target[targetOffset + 9] = (code.Code & (ushort) 0x1 << 6) == 0 ? falseValue : trueValue;
            target[targetOffset + 10] = (code.Code & (ushort) 0x1 << 5) == 0 ? falseValue : trueValue;
            target[targetOffset + 11] = (code.Code & (ushort) 0x1 << 4) == 0 ? falseValue : trueValue;
            target[targetOffset + 12] = (code.Code & (ushort) 0x1 << 3) == 0 ? falseValue : trueValue;
            target[targetOffset + 13] = (code.Code & (ushort) 0x1 << 2) == 0 ? falseValue : trueValue;
            target[targetOffset + 14] = (code.Code & (ushort) 0x1 << 1) == 0 ? falseValue : trueValue;
            target[targetOffset + 15] = (code.Code & (ushort) 0x1) == 0 ? falseValue : trueValue;
        }

        public static void SetBits<T>(GrayCodeU16 code, T[,] target, int targetRow, int targetOffset, T trueValue,
            T falseValue)
        {
            target[targetRow, targetOffset + 0] = (code.Code & (ushort) 0x1 << 15) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 1] = (code.Code & (ushort) 0x1 << 14) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 2] = (code.Code & (ushort) 0x1 << 13) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 3] = (code.Code & (ushort) 0x1 << 12) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 4] = (code.Code & (ushort) 0x1 << 11) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 5] = (code.Code & (ushort) 0x1 << 10) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 6] = (code.Code & (ushort) 0x1 << 9) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 7] = (code.Code & (ushort) 0x1 << 8) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 8] = (code.Code & (ushort) 0x1 << 7) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 9] = (code.Code & (ushort) 0x1 << 6) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 10] = (code.Code & (ushort) 0x1 << 5) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 11] = (code.Code & (ushort) 0x1 << 4) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 12] = (code.Code & (ushort) 0x1 << 3) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 13] = (code.Code & (ushort) 0x1 << 2) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 14] = (code.Code & (ushort) 0x1 << 1) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 15] = (code.Code & (ushort) 0x1) == 0 ? falseValue : trueValue;
        }

        public static T[] GetSetBits<T>(GrayCodeU16 code, T trueValue, T falseValue)
        {
            var ret = new T[16];
            SetBits(code, ret, 0, trueValue, falseValue);
            return ret;
        }

        public static ushort Decode(GrayCodeU16 code)
        {
            ushort mask;
            ushort num = code.Code;
            for (mask = (ushort) (num >> 1); mask != 0; mask = (ushort) (mask >> 1))
            {
                num = (ushort) (num ^ mask);
            }
            return num;
        }

        public static ushort Decode(ushort code)
        {
            var c = new GrayCodeU16 {Code = code};
            return Decode(c);
        }

        public static GrayCodeU16 Encode(ushort binary)
        {
            return new GrayCodeU16(binary);
        }

        public static explicit operator GrayCodeU16(ushort num)
        {
            return new GrayCodeU16(num);
        }

        public static explicit operator ushort(GrayCodeU16 code)
        {
            return Decode(code);
        }
    }

    public struct GrayCodeU32
    {
        private GrayCodeU32(uint binary)
            : this()
        {
            Code = (binary >> 1) ^ binary;
        }

        public uint Code { get; private set; }

        public static string DebugString(GrayCodeU32 code)
        {
            return new string(GetSetBits(code, '1', '0'));
        }

        public static void SetBits<T>(GrayCodeU32 code, T[] target, int targetOffset, T trueValue, T falseValue)
        {
            target[targetOffset + 0] = (code.Code & 0x1u << 31) == 0 ? falseValue : trueValue;
            target[targetOffset + 1] = (code.Code & 0x1u << 30) == 0 ? falseValue : trueValue;
            target[targetOffset + 2] = (code.Code & 0x1u << 29) == 0 ? falseValue : trueValue;
            target[targetOffset + 3] = (code.Code & 0x1u << 28) == 0 ? falseValue : trueValue;
            target[targetOffset + 4] = (code.Code & 0x1u << 27) == 0 ? falseValue : trueValue;
            target[targetOffset + 5] = (code.Code & 0x1u << 26) == 0 ? falseValue : trueValue;
            target[targetOffset + 6] = (code.Code & 0x1u << 25) == 0 ? falseValue : trueValue;
            target[targetOffset + 7] = (code.Code & 0x1u << 24) == 0 ? falseValue : trueValue;
            target[targetOffset + 8] = (code.Code & 0x1u << 23) == 0 ? falseValue : trueValue;
            target[targetOffset + 9] = (code.Code & 0x1u << 22) == 0 ? falseValue : trueValue;
            target[targetOffset + 10] = (code.Code & 0x1u << 21) == 0 ? falseValue : trueValue;
            target[targetOffset + 11] = (code.Code & 0x1u << 20) == 0 ? falseValue : trueValue;
            target[targetOffset + 12] = (code.Code & 0x1u << 19) == 0 ? falseValue : trueValue;
            target[targetOffset + 13] = (code.Code & 0x1u << 18) == 0 ? falseValue : trueValue;
            target[targetOffset + 14] = (code.Code & 0x1u << 17) == 0 ? falseValue : trueValue;
            target[targetOffset + 15] = (code.Code & 0x1u << 16) == 0 ? falseValue : trueValue;
            target[targetOffset + 16] = (code.Code & 0x1u << 15) == 0 ? falseValue : trueValue;
            target[targetOffset + 17] = (code.Code & 0x1u << 14) == 0 ? falseValue : trueValue;
            target[targetOffset + 18] = (code.Code & 0x1u << 13) == 0 ? falseValue : trueValue;
            target[targetOffset + 19] = (code.Code & 0x1u << 12) == 0 ? falseValue : trueValue;
            target[targetOffset + 20] = (code.Code & 0x1u << 11) == 0 ? falseValue : trueValue;
            target[targetOffset + 21] = (code.Code & 0x1u << 10) == 0 ? falseValue : trueValue;
            target[targetOffset + 22] = (code.Code & 0x1u << 9) == 0 ? falseValue : trueValue;
            target[targetOffset + 23] = (code.Code & 0x1u << 8) == 0 ? falseValue : trueValue;
            target[targetOffset + 24] = (code.Code & 0x1u << 7) == 0 ? falseValue : trueValue;
            target[targetOffset + 25] = (code.Code & 0x1u << 6) == 0 ? falseValue : trueValue;
            target[targetOffset + 26] = (code.Code & 0x1u << 5) == 0 ? falseValue : trueValue;
            target[targetOffset + 27] = (code.Code & 0x1u << 4) == 0 ? falseValue : trueValue;
            target[targetOffset + 28] = (code.Code & 0x1u << 3) == 0 ? falseValue : trueValue;
            target[targetOffset + 29] = (code.Code & 0x1u << 2) == 0 ? falseValue : trueValue;
            target[targetOffset + 30] = (code.Code & 0x1u << 1) == 0 ? falseValue : trueValue;
            target[targetOffset + 31] = (code.Code & 0x1u) == 0 ? falseValue : trueValue;
        }

        public static void SetBits<T>(GrayCodeU32 code, T[,] target, int targetRow, int targetOffset, T trueValue,
            T falseValue)
        {
            target[targetRow, targetOffset + 0] = (code.Code & 0x1u << 31) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 1] = (code.Code & 0x1u << 30) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 2] = (code.Code & 0x1u << 29) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 3] = (code.Code & 0x1u << 28) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 4] = (code.Code & 0x1u << 27) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 5] = (code.Code & 0x1u << 26) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 6] = (code.Code & 0x1u << 25) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 7] = (code.Code & 0x1u << 24) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 8] = (code.Code & 0x1u << 23) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 9] = (code.Code & 0x1u << 22) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 10] = (code.Code & 0x1u << 21) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 11] = (code.Code & 0x1u << 20) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 12] = (code.Code & 0x1u << 19) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 13] = (code.Code & 0x1u << 18) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 14] = (code.Code & 0x1u << 17) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 15] = (code.Code & 0x1u << 16) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 16] = (code.Code & 0x1u << 15) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 17] = (code.Code & 0x1u << 14) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 18] = (code.Code & 0x1u << 13) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 19] = (code.Code & 0x1u << 12) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 20] = (code.Code & 0x1u << 11) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 21] = (code.Code & 0x1u << 10) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 22] = (code.Code & 0x1u << 9) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 23] = (code.Code & 0x1u << 8) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 24] = (code.Code & 0x1u << 7) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 25] = (code.Code & 0x1u << 6) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 26] = (code.Code & 0x1u << 5) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 27] = (code.Code & 0x1u << 4) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 28] = (code.Code & 0x1u << 3) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 29] = (code.Code & 0x1u << 2) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 30] = (code.Code & 0x1u << 1) == 0 ? falseValue : trueValue;
            target[targetRow, targetOffset + 31] = (code.Code & 0x1u) == 0 ? falseValue : trueValue;
        }

        public static T[] GetSetBits<T>(GrayCodeU32 code, T trueValue, T falseValue)
        {
            var ret = new T[32];
            SetBits(code, ret, 0, trueValue, falseValue);
            return ret;
        }

        public static uint Decode(GrayCodeU32 code)
        {
            uint mask;
            uint num = code.Code;
            for (mask = num >> 1; mask != 0; mask = mask >> 1)
            {
                num = num ^ mask;
            }
            return num;
        }

        public static uint Decode(uint code)
        {
            var c = new GrayCodeU32 {Code = code};
            return Decode(c);
        }

        public static GrayCodeU32 Encode(uint binary)
        {
            return new GrayCodeU32(binary);
        }

        public static explicit operator GrayCodeU32(uint num)
        {
            return new GrayCodeU32(num);
        }

        public static explicit operator uint(GrayCodeU32 code)
        {
            return Decode(code);
        }
    }

    public struct GrayCodeU64
    {
        private GrayCodeU64(ulong binary)
            : this()
        {
            Code = (binary >> 1) ^ binary;
        }

        public ulong Code { get; private set; }

        public static string DebugString(GrayCodeU64 code)
        {
            var sb = new StringBuilder();
            int sz = 64;
            for (int i = sz - 1; i > -1; i--)
            {
                ulong mask = 0x1u << i;

                sb.Append((code.Code & mask) == 0 ? "0" : "1");
            }
            return sb.ToString();
        }

        public static T[] GetSetBits<T>(GrayCodeU64 code, T trueValue, T falseValue)
        {
            var ret = new T[64];
            throw new NotImplementedException();
        }

        public static ulong Decode(GrayCodeU64 code)
        {
            ulong mask;
            ulong num = code.Code;
            for (mask = num >> 1; mask != 0; mask = mask >> 1)
            {
                num = num ^ mask;
            }
            return num;
        }

        public static ulong Decode(ulong code)
        {
            var c = new GrayCodeU64 {Code = code};
            return Decode(c);
        }

        public static GrayCodeU64 Encode(ulong binary)
        {
            return new GrayCodeU64(binary);
        }

        public static explicit operator GrayCodeU64(ulong num)
        {
            return new GrayCodeU64(num);
        }

        public static explicit operator ulong(GrayCodeU64 code)
        {
            return Decode(code);
        }
    }
}