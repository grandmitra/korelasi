import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. CONFIG & AUTHENTICATION ---
st.set_page_config(page_title="GM Intelligence Dashboard", layout="wide")

def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "mbg212":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("""
            <div style='text-align: center; padding: 20px;'>
                <h2 style='color: #1E3A8A;'>🏛️ Grand Mitra Intelligence</h2>
                <p>Silakan masukkan password akses untuk melanjutkan</p>
            </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.text_input("Access Password", type="password", on_change=password_entered, key="password")
            if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                st.error("😕 Password salah. Silakan hubungi Administrator.")
        return False
    return True

if not check_password():
    st.stop()

# --- 2. DATA SOURCE CONFIG ---
JSON_KEY_FILE = 'KUNCI_AKSES.json' 
SHEET_ID = "1wI0htLSwlrrcOMDTx8QqL-7BAtRwkIJPATrsUxyfEaQ"
CACHE_DIR = "api_parquet_cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- 3. CACHE ENGINE (HYBRID AUTH) ---
@st.cache_data(ttl=3600)
def get_data_cached(sheet_name):
    cache_path = os.path.join(CACHE_DIR, f"{sheet_name}.parquet")
    if not os.path.exists(cache_path):
        scope = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        
        # Cek apakah ada file lokal, jika tidak ambil dari Streamlit Secrets (GitHub)
        if os.path.exists(JSON_KEY_FILE):
            creds = Credentials.from_service_account_file(JSON_KEY_FILE, scopes=scope)
        elif "gcp_service_account" in st.secrets:
            creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        else:
            st.error("Kunci Akses tidak ditemukan (JSON atau Secrets missing).")
            st.stop()
            
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SHEET_ID)
        worksheet = spreadsheet.worksheet(sheet_name)
        df = pd.DataFrame(worksheet.get_all_records())
        df = df.astype(str).replace(['nan', 'None', ''], 'Unknown').drop_duplicates()
        df.to_parquet(cache_path, engine='pyarrow', index=False)
        return df
    return pd.read_parquet(cache_path)

# --- 4. CORE PROCESSING ---
try:
    with st.spinner('Menganalisa Data Performa...'):
        df_sales_raw = get_data_cached("PENJUALAN")
        df_hpp_raw = get_data_cached("HPP")
        df_master_raw = get_data_cached("PRODUK_MASTER")
        df_stok_raw = get_data_cached("STOK")

    # A. Cleaning & Tipe Harga
    df_sales = df_sales_raw.copy()
    df_sales['FORM_DATE'] = pd.to_datetime(df_sales['FORM_DATE'], errors='coerce')
    df_sales['TAHUN'] = df_sales['FORM_DATE'].dt.year.fillna(0).astype(int).astype(str)
    df_sales['BULAN'] = df_sales['FORM_DATE'].dt.month_name().fillna('Unknown')
    
    for col in ['NET_AMOUNT', 'QTY']:
        df_sales[col] = pd.to_numeric(df_sales[col], errors='coerce').fillna(0)

    df_sales['TIPE_HARGA'] = df_sales['DISCOUNT_NO'].apply(
        lambda x: 'Harga Promo' if str(x) != 'Unknown' and str(x).strip() != '' else 'Harga Reguler'
    )
    df_sales['IS_PROMO'] = df_sales['TIPE_HARGA'].apply(lambda x: 1 if x == 'Harga Promo' else 0)

    # HPP & Master
    df_hpp_clean = df_hpp_raw.drop_duplicates(subset=['ITEM_NO'], keep='last').copy()
    df_hpp_clean['ITEM_NO'] = df_hpp_clean['ITEM_NO'].astype(str)
    df_hpp_clean['HPP'] = pd.to_numeric(df_hpp_clean['HPP'], errors='coerce').fillna(0)

    df_sales['MY'] = df_sales['FORM_DATE'].dt.to_period('M')
    total_months = max(df_sales['MY'].nunique(), 1)
    
    # Intelligence Mapping
    cons = df_sales.groupby('ITEM_NO')['MY'].nunique().reset_index()
    def cat_logic(m):
        pct = (m / total_months) * 100
        if m == 1: return "🏗️ PROJECT"
        elif pct >= 80: return "💎 BASIC"
        elif 50 <= pct < 80: return "📦 REGULER"
        else: return "🔍 OTHERS"
    cons['KATEGORI'] = cons['MY'].apply(cat_logic)

    avg_s = (df_sales.groupby('ITEM_NO')['QTY'].sum() / total_months).reset_index()
    def fsd_logic(v):
        if v > 50: return "🚀 FAST"
        elif v > 5: return "🐢 SLOW"
        else: return "💀 DEAD"
    avg_s['FSD'] = avg_s['QTY'].apply(fsd_logic)

    df_master_raw['ITEM_NO'] = df_master_raw['ITEM_NO'].astype(str)
    df_master_clean = df_master_raw[['ITEM_NO', 'ITEM_NAME', 'GROUP_NAME1', 'GROUP_NAME2', 'GROUP_NAME3', 'GROUP_NAME4', 'VENDOR_NAME']].drop_duplicates(subset=['ITEM_NO'])

    df_intel = pd.merge(df_master_clean, cons[['ITEM_NO', 'KATEGORI']], on='ITEM_NO', how='left')
    df_intel = pd.merge(df_intel, avg_s[['ITEM_NO', 'FSD', 'QTY']], on='ITEM_NO', how='left')
    
    df_final = pd.merge(df_sales, df_intel, on='ITEM_NO', how='left', suffixes=('', '_dup'))
    df_final = df_final.loc[:, ~df_final.columns.str.contains('_dup')]
    df_final = pd.merge(df_final, df_hpp_clean[['ITEM_NO', 'HPP']], on='ITEM_NO', how='left')
    df_final['PROFIT'] = df_final['NET_AMOUNT'] - (df_final['QTY'] * df_final['HPP'])

    # Stok Valuasi
    df_stok_clean = df_stok_raw.copy()
    df_stok_clean['ITEM_NO'] = df_stok_clean['ITEM_NO'].astype(str)
    df_stok_clean['BALANCE_QTY'] = pd.to_numeric(df_stok_clean['BALANCE_QTY'], errors='coerce').fillna(0)
    df_stok_val = pd.merge(df_stok_clean, df_intel, on='ITEM_NO', how='left', suffixes=('', '_dup'))
    df_stok_val = pd.merge(df_stok_val, df_hpp_clean[['ITEM_NO', 'HPP']], on='ITEM_NO', how='left')
    df_stok_val['STOCK_RUPIAH'] = df_stok_val['BALANCE_QTY'] * df_stok_val['HPP'].fillna(0)

    # --- 5. SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Global Filters")
    all_item_list = sorted([str(x) for x in df_intel['ITEM_NAME'].unique().tolist()])
    search_item = st.sidebar.selectbox("🔎 Cari Nama Barang", ["Semua Barang"] + all_item_list)

    def multi_filter(label, col, df):
        options = sorted([str(x) for x in df[col].unique().tolist() if str(x) != 'nan'])
        return st.sidebar.multiselect(label, options)

    f_thrg = multi_filter("TIPE HARGA", 'TIPE_HARGA', df_final)
    f_tahun = multi_filter("TAHUN", 'TAHUN', df_final)
    f_bulan = multi_filter("BULAN", 'BULAN', df_final)
    f_vend = multi_filter("VENDOR NAME", 'VENDOR_NAME', df_final)
    f_kat = multi_filter("KATEGORI", 'KATEGORI', df_final)
    f_fsd = multi_filter("FSD", 'FSD', df_final)
    f_g1 = multi_filter("GROUP 1 (Dept)", 'GROUP_NAME1', df_final)

    # Filtering Logic
    df_f = df_final.copy()
    if search_item != "Semua Barang": df_f = df_f[df_f['ITEM_NAME'] == search_item]
    if f_thrg: df_f = df_f[df_f['TIPE_HARGA'].isin(f_thrg)]
    if f_tahun: df_f = df_f[df_f['TAHUN'].isin(f_tahun)]
    if f_bulan: df_f = df_f[df_f['BULAN'].isin(f_bulan)]
    if f_vend: df_f = df_f[df_f['VENDOR_NAME'].isin(f_vend)]
    if f_kat: df_f = df_f[df_f['KATEGORI'].isin(f_kat)]
    if f_fsd: df_f = df_f[df_f['FSD'].isin(f_fsd)]
    if f_g1: df_f = df_f[df_f['GROUP_NAME1'].isin(f_g1)]

    df_s_f = df_stok_val.copy()
    if f_vend: df_s_f = df_s_f[df_s_f['VENDOR_NAME'].isin(f_vend)]
    if f_kat: df_s_f = df_s_f[df_s_f['KATEGORI'].isin(f_kat)]
    if f_fsd: df_s_f = df_s_f[df_s_f['FSD'].isin(f_fsd)]
    if f_g1: df_s_f = df_s_f[df_s_f['GROUP_NAME1'].isin(f_g1)]

    # --- 6. MAIN UI ---
    st.title("📊 Grand Mitra Intelligence Dashboard")
    
    rev, prof = df_f['NET_AMOUNT'].sum(), df_f['PROFIT'].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"Rp {rev:,.0f}")
    c2.metric("Total Margin", f"Rp {prof:,.0f}")
    c3.metric("Margin (%)", f"{(prof/rev*100 if rev>0 else 0):.2f}%")
    c4.metric("Stock Value", f"Rp {df_s_f['STOCK_RUPIAH'].sum():,.0f}")

    st.divider()
    tabs = st.tabs(["📋 Resume Performa", "📈 Trend Penjualan", "📦 Daftar SOQ", "🧪 Bedah Korelasi Dept", "🏆 Top 5 Promo Items"])

    with tabs[0]: # TAB RESUME
        col_t1, col_t2 = st.columns([0.7, 0.3])
        df_res_g1 = df_f.groupby('GROUP_NAME1').agg({
            'NET_AMOUNT': 'sum', 'PROFIT': 'sum', 'QTY': 'sum', 'ITEM_NO': 'nunique'
        }).reset_index().rename(columns={'ITEM_NO': 'SKU_Terjual', 'QTY': 'Qty_Terjual'})
        df_inv_g1 = df_s_f[df_s_f['BALANCE_QTY'] > 0].groupby('GROUP_NAME1').agg({
            'STOCK_RUPIAH': 'sum', 'ITEM_NO': 'nunique'
        }).reset_index().rename(columns={'ITEM_NO': 'SKU_Stok_Aktif'})
        df_resume = pd.merge(df_res_g1, df_inv_g1, on='GROUP_NAME1', how='outer').fillna(0)
        df_resume['SSR'] = (df_resume['STOCK_RUPIAH'] / df_resume['NET_AMOUNT']).replace([float('inf')], 0).fillna(0)
        df_resume['CTS_%'] = (df_resume['NET_AMOUNT'] / rev * 100) if rev > 0 else 0
        df_resume['GM_%'] = (df_resume['PROFIT'] / df_resume['NET_AMOUNT'] * 100).replace([float('inf')], 0).fillna(0)

        with col_t1:
            st.subheader("Departmental Performance Matrix")
            st.dataframe(df_resume.sort_values('NET_AMOUNT', ascending=False).style.format({
                'NET_AMOUNT': 'Rp {:,.0f}', 'PROFIT': 'Rp {:,.0f}', 'STOCK_RUPIAH': 'Rp {:,.0f}', 
                'SSR': '{:.2f}x', 'CTS_%': '{:.2f}%', 'GM_%': '{:.2f}%'
            }), use_container_width=True)
        with col_t2:
            st.subheader("CTS % Share")
            fig_pie = px.pie(df_resume, values='NET_AMOUNT', names='GROUP_NAME1', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    with tabs[1]: # TAB TREND
        df_tl = df_f.groupby('MY').agg({'NET_AMOUNT': 'sum', 'PROFIT': 'sum'}).reset_index()
        df_tl['MY_STR'] = df_tl['MY'].astype(str)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=df_tl['MY_STR'], y=df_tl['NET_AMOUNT'], name="Revenue"))
        fig_trend.add_trace(go.Scatter(x=df_tl['MY_STR'], y=df_tl['PROFIT'], name="Margin", line=dict(color='green')))
        st.plotly_chart(fig_trend, use_container_width=True)

    with tabs[2]: # TAB SOQ
        # Sederhana: SOQ = (Rata2 Sales * 3) - Stok Saat Ini
        df_sales_avg = df_f.groupby('ITEM_NO')['QTY'].sum() / total_months
        df_soq = pd.merge(df_s_f, df_sales_avg.reset_index().rename(columns={'QTY': 'AVG_QTY'}), on='ITEM_NO', how='left')
        df_soq['SOQ'] = (df_soq['AVG_QTY'] * 3) - df_soq['BALANCE_QTY']
        st.dataframe(df_soq[df_soq['SOQ'] > 0][['ITEM_NO', 'ITEM_NAME', 'BALANCE_QTY', 'SOQ']].sort_values('SOQ', ascending=False), use_container_width=True)

    with tabs[3]: # TAB BEDAH KORELASI
        st.subheader("🧪 Analisis Kedalaman Efektivitas Promo")
        df_stat = df_f.copy()
        
        g_corr = df_stat['IS_PROMO'].corr(df_stat['NET_AMOUNT']) if df_stat['IS_PROMO'].nunique() > 1 else 0
        g_avg_r = df_stat[df_stat['IS_PROMO'] == 0]['NET_AMOUNT'].mean()
        g_avg_p = df_stat[df_stat['IS_PROMO'] == 1]['NET_AMOUNT'].mean()
        g_lift = ((g_avg_p - g_avg_r) / g_avg_r * 100) if g_avg_r > 0 else 0

        # Dinamis Strategic Insight
        if g_lift > 200 and g_corr < 0.2:
            status, s_what, s_why = "OUTLIER DETECTED", "Audit Transaksi Proyek", f"Lompatan omzet sangat tinggi ({g_lift:.1f}%) tapi korelasi rendah. Promo hanya dimanfaatkan transaksi borongan."
        elif g_corr > 0.3:
            status, s_what, s_why = "PROMO SEHAT", "Ekspansi Program Loyalitas", "Pelanggan merespon promo secara konsisten dan merata."
        else:
            status, s_what, s_why = "MODERATE IMPACT", "Review Display & Signage", "Promo memberikan dampak namun belum cukup kuat menarik nilai transaksi."

        st.info(f"**Analisa Global Toko:** Pearson: **{g_corr:.4f}** | Lompatan: **{g_lift:.2f}%** | Status: **{status}**")
        
        st.subheader("💡 Strategic Action Plan (5W+1H)")
        strat_data = {
            "Point": ["What", "Where", "Why", "Who", "Whom", "How"],
            "Strategic Action": [s_what, "Area Sales & Digital", s_why, "Pak Sofyan (CM)", "Retail & B2B", "Cek ketersediaan Top 5 Item Promo."]
        }
        st.table(pd.DataFrame(strat_data))
        
        # Bubble Chart per Dept
        dept_data = []
        for dpt in df_stat['GROUP_NAME1'].unique():
            sub = df_stat[df_stat['GROUP_NAME1'] == dpt]
            if len(sub) > 2 and sub['IS_PROMO'].nunique() > 1:
                c = sub['IS_PROMO'].corr(sub['NET_AMOUNT'])
                lr = sub[sub['IS_PROMO'] == 0]['NET_AMOUNT'].mean()
                lp = sub[sub['IS_PROMO'] == 1]['NET_AMOUNT'].mean()
                lf = ((lp - lr) / lr * 100) if lr > 0 else 0
                dept_data.append({'Dept': dpt, 'Pearson': c, 'Lift%': lf, 'Trx': len(sub)})
        
        if dept_data:
            fig_b = px.scatter(pd.DataFrame(dept_data), x="Pearson", y="Lift%", size="Trx", color="Dept", text="Dept")
            st.plotly_chart(fig_b, use_container_width=True)

    with tabs[4]: # TOP 5 PROMO
        df_p = df_f[df_f['IS_PROMO'] == 1].copy()
        if not df_p.empty:
            top_p = df_p.groupby(['GROUP_NAME1', 'ITEM_NAME']).agg({'NET_AMOUNT': 'sum', 'QTY': 'sum'}).reset_index()
            top_p = top_p.sort_values(['GROUP_NAME1', 'NET_AMOUNT'], ascending=[True, False]).groupby('GROUP_NAME1').head(5)
            st.dataframe(top_p.style.format({'NET_AMOUNT': 'Rp {:,.0f}', 'QTY': '{:,.0f}'}), use_container_width=True)

except Exception as e:
    st.error(f"Gagal memproses data: {e}")
