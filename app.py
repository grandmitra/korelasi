import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. CONFIG & UI ---
st.set_page_config(page_title="GM Intelligence Dashboard", layout="wide")

# --- SISTEM LOGIN (PASSWORD: mbg212) ---
def check_password():
    def password_entered():
        if st.session_state["password_input"] == "mbg212":
            st.session_state["password_correct"] = True
            del st.session_state["password_input"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<h2 style='text-align: center;'>🏛️ Grand Mitra Intelligence</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1.5,1])
        with col2:
            st.text_input("Masukkan Password Akses", type="password", on_change=password_entered, key="password_input")
        return False
    elif not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1,1.5,1])
        with col2:
            st.text_input("Masukkan Password Akses", type="password", on_change=password_entered, key="password_input")
            st.error("😕 Password salah.")
        return False
    return True

if not check_password():
    st.stop()

# --- KONFIGURASI AKSES ---
JSON_KEY_FILE = 'KUNCI_AKSES.json' 
SHEET_ID = "1wI0htLSwlrrcOMDTx8QqL-7BAtRwkIJPATrsUxyfEaQ"
CACHE_DIR = "api_parquet_cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- 2. CACHE ENGINE ---
@st.cache_data(ttl=3600)
def get_data_cached(sheet_name):
    cache_path = os.path.join(CACHE_DIR, f"{sheet_name}.parquet")
    # Jika file cache tidak ada, ambil dari Google Sheets
    if not os.path.exists(cache_path):
        scope = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        try:
            creds = Credentials.from_service_account_file(JSON_KEY_FILE, scopes=scope)
            client = gspread.authorize(creds)
            spreadsheet = client.open_by_key(SHEET_ID)
            worksheet = spreadsheet.worksheet(sheet_name)
            df = pd.DataFrame(worksheet.get_all_records())
            
            # Cleaning dasar
            df = df.astype(str).replace(['nan', 'None', ''], 'Unknown').drop_duplicates()
            
            # Simpan ke Parquet untuk pembacaan cepat berikutnya
            df.to_parquet(cache_path, engine='pyarrow', index=False)
            return df
        except Exception as e:
            st.error(f"Gagal mengambil data {sheet_name}: {e}")
            return pd.DataFrame()
    return pd.read_parquet(cache_path)

# --- 3. CORE PROCESSING ---
try:
    with st.spinner('Sinkronisasi Data & Analisis Strategis...'):
        df_sales_raw = get_data_cached("PENJUALAN")
        df_hpp_raw = get_data_cached("HPP")
        df_master_raw = get_data_cached("PRODUK_MASTER")
        df_stok_raw = get_data_cached("STOK")

    # A. Penjualan Cleaning
    df_sales = df_sales_raw.copy()
    df_sales['FORM_DATE'] = pd.to_datetime(df_sales['FORM_DATE'], errors='coerce')
    df_sales['TAHUN'] = df_sales['FORM_DATE'].dt.year.fillna(0).astype(int).astype(str)
    df_sales['BULAN'] = df_sales['FORM_DATE'].dt.month_name().fillna('Unknown')
    
    for col in ['NET_AMOUNT', 'QTY']:
        df_sales[col] = pd.to_numeric(df_sales[col], errors='coerce').fillna(0)

    # Logika Promo
    df_sales['TIPE_HARGA'] = df_sales['DISCOUNT_NO'].apply(
        lambda x: 'Harga Promo' if str(x) != 'Unknown' and str(x).strip() != '' else 'Harga Reguler'
    )
    df_sales['IS_PROMO'] = df_sales['TIPE_HARGA'].apply(lambda x: 1 if x == 'Harga Promo' else 0)

    # B. HPP & Master Cleaning
    df_hpp_clean = df_hpp_raw.drop_duplicates(subset=['ITEM_NO'], keep='last').copy()
    df_hpp_clean['ITEM_NO'] = df_hpp_clean['ITEM_NO'].astype(str)
    df_hpp_clean['HPP'] = pd.to_numeric(df_hpp_clean['HPP'], errors='coerce').fillna(0)

    # C. Intelligence Mapping (Kategori & FSD)
    df_sales['MY'] = df_sales['FORM_DATE'].dt.to_period('M')
    total_months = max(df_sales['MY'].nunique(), 1)
    
    # Konsistensi Penjualan (Kategori)
    cons = df_sales.groupby('ITEM_NO')['MY'].nunique().reset_index()
    def cat_logic(m):
        pct = (m / total_months) * 100
        if m == 1: return "🏗️ PROJECT"
        elif pct >= 80: return "💎 BASIC"
        elif 50 <= pct < 80: return "📦 REGULER"
        else: return "🔍 OTHERS"
    cons['KATEGORI'] = cons['MY'].apply(cat_logic)

    # Kecepatan Penjualan (FSD)
    avg_s = (df_sales.groupby('ITEM_NO')['QTY'].sum() / total_months).reset_index()
    def fsd_logic(v):
        if v > 50: return "🚀 FAST"
        elif v > 5: return "🐢 SLOW"
        else: return "💀 DEAD"
    avg_s['FSD'] = avg_s['QTY'].apply(fsd_logic)

    # Join Master Data
    df_master_clean = df_master_raw[['ITEM_NO', 'ITEM_NAME', 'GROUP_NAME1', 'GROUP_NAME2', 'GROUP_NAME3', 'GROUP_NAME4', 'VENDOR_NAME', 'ON_CONSIGNMENT']].drop_duplicates(subset=['ITEM_NO'])
    
    df_intel = pd.merge(df_master_clean, cons[['ITEM_NO', 'KATEGORI']], on='ITEM_NO', how='left')
    df_intel = pd.merge(df_intel, avg_s[['ITEM_NO', 'FSD', 'QTY']], on='ITEM_NO', how='left')
    df_intel['AVG_SALES'] = pd.to_numeric(df_intel['QTY'], errors='coerce').fillna(0)

    # Final Sales DataFrame
    df_final = pd.merge(df_sales, df_intel, on='ITEM_NO', how='left', suffixes=('', '_dup'))
    df_final = df_final.loc[:, ~df_final.columns.str.contains('_dup')]
    df_final = pd.merge(df_final, df_hpp_clean[['ITEM_NO', 'HPP']], on='ITEM_NO', how='left')
    df_final['PROFIT'] = df_final['NET_AMOUNT'] - (df_final['QTY'] * df_final['HPP'])

    # D. Stok DataFrame
    df_stok_clean = df_stok_raw.copy()
    df_stok_clean['ITEM_NO'] = df_stok_clean['ITEM_NO'].astype(str)
    df_stok_clean['BALANCE_QTY'] = pd.to_numeric(df_stok_clean['BALANCE_QTY'], errors='coerce').fillna(0)
    df_stok_val = pd.merge(df_stok_clean, df_intel, on='ITEM_NO', how='left', suffixes=('', '_dup'))
    df_stok_val = df_stok_val.loc[:, ~df_stok_val.columns.str.contains('_dup')]
    df_stok_val = pd.merge(df_stok_val, df_hpp_clean[['ITEM_NO', 'HPP']], on='ITEM_NO', how='left')
    df_stok_val['STOCK_RUPIAH'] = df_stok_val['BALANCE_QTY'] * df_stok_val['HPP'].fillna(0)

    # --- 4. SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Global Filters")
    
    # Search Box
    all_item_names = sorted([str(x) for x in df_intel['ITEM_NAME'].unique().tolist()])
    search_item = st.sidebar.selectbox("🔎 Cari Nama Barang", ["Semua Barang"] + all_item_names)

    # Helper Filter Function
    def multi_filter(label, col, df):
        options = sorted([str(x) for x in df[col].unique().tolist() if str(x) != 'nan'])
        return st.sidebar.multiselect(label, options)

    # Filter Groupings
    f_consign = multi_filter("🏷️ STATUS KONSINYASI", 'ON_CONSIGNMENT', df_final)
    f_thrg = multi_filter("💰 TIPE HARGA", 'TIPE_HARGA', df_final)
    f_tahun = multi_filter("📅 TAHUN", 'TAHUN', df_final)
    f_bulan = multi_filter("📅 BULAN", 'BULAN', df_final)
    f_vend = multi_filter("🏭 VENDOR", 'VENDOR_NAME', df_final)
    
    st.sidebar.divider()
    f_kat = multi_filter("📊 KATEGORI", 'KATEGORI', df_final)
    f_fsd = multi_filter("⚡ FSD STATUS", 'FSD', df_final)
    
    st.sidebar.divider()
    f_g1 = multi_filter("📦 GROUP 1 (Dept)", 'GROUP_NAME1', df_final)
    f_g2 = multi_filter("📦 GROUP 2", 'GROUP_NAME2', df_final)

    # Apply Logic Filtering
    def apply_filters(df, is_stok=False):
        dff = df.copy()
        if not is_stok: # Filter khusus penjualan
            if f_thrg: dff = dff[dff['TIPE_HARGA'].isin(f_thrg)]
            if f_tahun: dff = dff[dff['TAHUN'].isin(f_tahun)]
            if f_bulan: dff = dff[dff['BULAN'].isin(f_bulan)]
        
        # Filter umum (ada di master/intel)
        if search_item != "Semua Barang": dff = dff[dff['ITEM_NAME'] == search_item]
        if f_consign: dff = dff[dff['ON_CONSIGNMENT'].isin(f_consign)]
        if f_vend: dff = dff[dff['VENDOR_NAME'].isin(f_vend)]
        if f_kat: dff = dff[dff['KATEGORI'].isin(f_kat)]
        if f_fsd: dff = dff[dff['FSD'].isin(f_fsd)]
        if f_g1: dff = dff[dff['GROUP_NAME1'].isin(f_g1)]
        if f_g2: dff = dff[dff['GROUP_NAME2'].isin(f_g2)]
        return dff

    df_f = apply_filters(df_final)
    df_s_f = apply_filters(df_stok_val, is_stok=True)

    # --- 5. MAIN UI ---
    st.title("📊 Grand Mitra Intelligence Dashboard")
    
    # Scorecards
    rev, prof = df_f['NET_AMOUNT'].sum(), df_f['PROFIT'].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"Rp {rev:,.0f}")
    c2.metric("Total Margin", f"Rp {prof:,.0f}")
    c3.metric("Margin (%)", f"{(prof/rev*100 if rev>0 else 0):.2f}%")
    c4.metric("Stock Value", f"Rp {df_s_f['STOCK_RUPIAH'].sum():,.0f}")

    st.divider()
    tabs = st.tabs(["📋 Resume Performa", "📈 Trend Penjualan", "📦 Daftar SOQ", "🧪 Bedah Promo", "🏆 Top 5 Items"])

    with tabs[0]: # TAB RESUME
        col_t1, col_t2 = st.columns([0.7, 0.3])
        
        # Agregasi Dept
        df_res_g1 = df_f.groupby('GROUP_NAME1').agg({
            'NET_AMOUNT': 'sum', 'PROFIT': 'sum', 'QTY': 'sum', 'ITEM_NO': 'nunique'
        }).reset_index().rename(columns={'ITEM_NO': 'SKU_Terjual', 'QTY': 'Qty_Terjual'})
        
        df_inv_g1 = df_s_f[df_s_f['BALANCE_QTY'] > 0].groupby('GROUP_NAME1').agg({
            'STOCK_RUPIAH': 'sum', 'ITEM_NO': 'nunique'
        }).reset_index().rename(columns={'ITEM_NO': 'SKU_Stok_Aktif'})
        
        df_resume = pd.merge(df_res_g1, df_inv_g1, on='GROUP_NAME1', how='outer').fillna(0)
        df_resume['SSR'] = (df_resume['STOCK_RUPIAH'] / df_resume['NET_AMOUNT']).replace([np.inf, -np.inf], 0).fillna(0)
        df_resume['CTS_%'] = (df_resume['NET_AMOUNT'] / rev * 100) if rev > 0 else 0
        df_resume['GM_%'] = (df_resume['PROFIT'] / df_resume['NET_AMOUNT'] * 100).replace([np.inf, -np.inf], 0).fillna(0)

        with col_t1:
            st.subheader("Departmental Performance Matrix")
            st.dataframe(df_resume.sort_values('NET_AMOUNT', ascending=False).style.format({
                'NET_AMOUNT': 'Rp {:,.0f}', 'PROFIT': 'Rp {:,.0f}', 'STOCK_RUPIAH': 'Rp {:,.0f}', 
                'SSR': '{:.2f}x', 'CTS_%': '{:.2f}%', 'GM_%': '{:.2f}%', 'SKU_Terjual': '{:,.0f}'
            }), use_container_width=True)

        with col_t2:
            st.subheader("Revenue Share")
            fig_pie = px.pie(df_resume, values='NET_AMOUNT', names='GROUP_NAME1', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    with tabs[1]: # TAB TREND
        df_tl = df_f.groupby('MY').agg({'NET_AMOUNT': 'sum', 'PROFIT': 'sum'}).reset_index()
        df_tl['MY_STR'] = df_tl['MY'].astype(str)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=df_tl['MY_STR'], y=df_tl['NET_AMOUNT'], name="Revenue", marker_color='#3498db'))
        fig_trend.add_trace(go.Scatter(x=df_tl['MY_STR'], y=df_tl['PROFIT'], name="Margin", line=dict(color='#2ecc71', width=3)))
        st.plotly_chart(fig_trend, use_container_width=True)

    with tabs[2]: # TAB SOQ (Suggested Order Qty)
        st.subheader("📦 Rekomendasi Order (Stok Dasar 3 Bulan)")
        df_sales_avg = df_f.groupby('ITEM_NO')['NET_AMOUNT'].sum() / total_months
        df_soq = pd.merge(df_s_f, df_sales_avg.reset_index().rename(columns={'NET_AMOUNT': 'AVG_RP'}), on='ITEM_NO', how='left')
        df_soq['SOQ'] = (df_soq['AVG_SALES'] * 3) - df_soq['BALANCE_QTY']
        
        df_soq_display = df_soq[df_soq['SOQ'] > 0][['ITEM_NO', 'ITEM_NAME', 'VENDOR_NAME', 'BALANCE_QTY', 'SOQ']].sort_values('SOQ', ascending=False)
        st.dataframe(df_soq_display, use_container_width=True)

    with tabs[3]: # TAB BEDAH PROMO
        st.subheader("🧪 Efektivitas Promo terhadap Omzet")
        df_stat = df_f.copy()
        if not df_stat.empty and df_stat['IS_PROMO'].nunique() > 1:
            g_corr = df_stat['IS_PROMO'].corr(df_stat['NET_AMOUNT'])
            st.info(f"Korelasi Global Promo vs Omzet: **{g_corr:.4f}**")
            
            fig_scatter = px.scatter(df_stat, x="QTY", y="NET_AMOUNT", color="TIPE_HARGA", 
                                     title="Sebaran Transaksi: Promo vs Reguler", trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Data tidak cukup untuk melakukan analisis korelasi (butuh perbandingan Harga Promo & Reguler).")

    with tabs[4]: # TAB TOP 5
        st.subheader("🏆 Produk Terlaris (Berdasarkan Filter)")
        top_5 = df_f.groupby(['GROUP_NAME1', 'ITEM_NAME']).agg({'NET_AMOUNT': 'sum', 'QTY': 'sum'}).reset_index()
        top_5 = top_5.sort_values(['GROUP_NAME1', 'NET_AMOUNT'], ascending=[True, False]).groupby('GROUP_NAME1').head(5)
        st.dataframe(top_5.style.format({'NET_AMOUNT': 'Rp {:,.0f}', 'QTY': '{:,.0f}'}), use_container_width=True)

except Exception as e:
    st.error(f"Terjadi kesalahan sistem: {e}")
    st.info("Saran: Coba hapus folder 'api_parquet_cache' dan jalankan ulang.")
