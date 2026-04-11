using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Drawing;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Spreadsheet;
using WinFormsApp_Draft;


namespace WinFormsApp_Draft.Auto
{
    public class ReadExcel
    {
        [DllImport("ReadExcelDLL.dll", EntryPoint = "parameters")]
        public static extern int parameters(int row, int col, string filepath, ref int data);

        public static List<int> row_param(int round, int data_points, string filepath)
        {
            List<int> result = new List<int>();
            int data = 0;

            for(int i = 0; i < data_points; i++)
            {
                int file_opened = parameters(round, i, filepath, ref data);
                if (file_opened == 1)
                {
                    result.Add(data);
                }
                else
                {
                    throw new Exception("Unable to open file");
                }   
            }
            return result;
        }

        public static List<int> col_param(int col, int data_points, string filepath)
        {
            List<int> result = new List<int>();
            int data = 0;

            for (int i = 1; i <= data_points; i++)
            {
                int file_opened = parameters(i, col, filepath, ref data);
                if (file_opened == 1)
                {
                    result.Add(data);
                }
                else
                {
                    throw new Exception("Unable to open file");
                }
            }
            return result;
        }
    }
}
