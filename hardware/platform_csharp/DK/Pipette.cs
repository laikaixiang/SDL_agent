using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace WinFormsApp_Draft.DK
{
    class Pipette
    {
        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Cpump_zero_p(byte index, byte id);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Find_zero_p(byte index, byte id);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Back_tip_p(byte index, byte id);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Find_tip_p(int index, byte id, ref byte return_value);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Find_status_p(int index, byte id, ref byte return_value);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Suck_p(byte index, byte id, Int16 suck_ul, ref byte return_value);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Spit_p(byte index, byte id, Int16 suck_ul, ref byte return_value);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Second_Suck_p(int index, byte id, ref byte return_value);
    }
}
