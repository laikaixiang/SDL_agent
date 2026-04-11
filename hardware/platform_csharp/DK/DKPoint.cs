using System;

namespace WinFormsApp_Draft.DK
{
    public class DKPoint
    {
        /// <summary>
        /// X 轴位置，0-30000
        /// </summary>
        public int x { get; set; }

        /// <summary>
        /// Y 轴位置，0-26500
        /// </summary>
        public int y { get; set; }

        /// <summary>
        /// 左滴头高度，0-150000
        /// </summary>
        public int lz { get; set; }

        /// <summary>
        /// 右滴头高度，0-150000
        /// </summary>
        public int rz { get; set; }

        public DKPoint() { }
        public DKPoint(int x, int y, int lz, int rz)
        {
            this.x = x;
            this.y = y;
            this.lz = lz;
            this.rz = rz;
        }

        public override string ToString()
        {
            string str = String.Format("{0},{1},{2},{3}",
                this.x, this.y, this.lz, this.rz);
            return str;
        }
    }
}
