# PDE-SOLVER
波動方程式的解的模擬
以有限差分 (Finite Difference Method, FDM)與傅立葉頻譜法 (Fourier Spectral Method, FFT) ，實作二維波動方程式 (2D Wave Equation) 的數值解，並支援多種邊界條件與座標系統。  
整體流程展示 數學建模 → 離散化 → Python 實作 → 視覺化 (GIF)。

---

## 功能特色
- **數值方法**：有限差分 (FDM)、傅立葉頻譜法 (FFT, 僅適用於週期邊界條件)
- **座標系統**：笛卡爾座標 (Cartesian)、圓柱座標 (Cylindrical)、球對稱 (Spherical radial)
- **邊界條件**：Dirichlet / Neumann / Periodic
- **視覺化**：可輸出 GIF，適合教學與展示
