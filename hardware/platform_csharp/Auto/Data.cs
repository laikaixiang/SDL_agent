using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace WinFormsApp_Draft.Auto
{
    public class Scan
    {
        public static void do_scan(Socket client)
        {
            Byte[] data = Encoding.Default.GetBytes("Scan\r");
            client.Send(data);
            Thread.Sleep(100);
        }

        /// <summary>
        /// control spectrometer to scan the substrate once and save data in its registers
        /// </summary>
        /// <param name="client"></param>
        public static async void do_single_scan(Socket client)
        {
            Byte[] data = Encoding.Default.GetBytes("scan1\r");
            client.Send(data);
            await Task.Delay(500);
        }
    }

    public class Data
    {
        private static void readDoubleArray(Socket client, ref double[] dbArray)
        {
            byte[] data = new byte[5000 * 8];
            client.Receive(data, 2, SocketFlags.None);
            int size = data[0] + data[1] * 256;
            client.Receive(data, size * 8, SocketFlags.None);

            IntPtr pnt = Marshal.AllocHGlobal(size * 8);

            try
            {
                //copy the array to unmanaged memory
                Marshal.Copy(data, 0, pnt, size * 8);
                //copy the unmanaged array back to another managed array
                dbArray = new double[size];
                Marshal.Copy(pnt, dbArray, 0, size);
            }
            finally
            {
                //free the unmanaged memory 
                Marshal.FreeHGlobal(pnt);
            }
        }

        /// <summary>
        /// read the results of the scanning process from spectrometer
        /// </summary>
        /// <param name="client"></param>
        /// <returns>
        /// 2 row list of wavelength and counts respectively
        /// </returns>
        public static List<string> get_data(Socket client)
        {
            client.ReceiveTimeout = 10000;
            double[] wl = null;
            double[] cts = null;
            //get wavelength
            Byte[] data = Encoding.Default.GetBytes("GetData/Type=wl\r");
            client.Send(data);
            Thread.Sleep(100);
            readDoubleArray(client, ref wl);
            //get counts
            data = Encoding.Default.GetBytes("GetData/Type=spec\r");
            client.Send(data);
            Thread.Sleep(100);
            readDoubleArray(client, ref cts);
            List<string> spectrum = new List<string>();
            string wavelength = null;
            string counts = null;

            for (int i = 0; i < wl.Length; i++)
            {
                wavelength += wl[i].ToString() + " ";
                counts += cts[i].ToString() + " ";
            }
            spectrum.Add(wavelength);
            spectrum.Add(counts);

            return spectrum;
        }
    }
}
