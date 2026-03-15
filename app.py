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

# --- 2. SISTEM LOGIN (PASSWORD: mbg212) ---
def check_password():
    """Memverifikasi password dengan proteksi session_state."""
    def password_entered():
        if st.session_state.get("password_input", "") == "mbg212":
            st.session_state["password_correct"] = True
            if "password_input" in st.session_state:
                del st.session_state["password_input"] 
        else:
            st.session_state["password_correct"] = False

    if not st.session_state.get("password_correct", False):
        st.markdown("<h2 style='text-align: center;'>🏛️ Grand Mitra Intelligence</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1.5,1])
        with col2:
            st.text_input("Masukkan Password Akses", type="password", on_change=password_entered, key="password_input")
            if st.session_state.get("password_correct") == False:
                st.error("😕 Password salah. Silakan coba lagi.")
        return False
    return True

if not check_password():
    st.stop()

# --- 3. DATA SOURCE CONFIG ---
JSON_KEY_FILE = 'KUNCI_AKSES.json' 
SHEET_ID = "1wI0htLSwlrrcOMDTx8QqL-7BAtRwkIJPATrsUxyfEaQ"
CACHE_DIR = "api_parquet_cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- 4. CACHE ENGINE ---
@st.cache_data(ttl=3600)
def get_data_cached(sheet_name):
    cache_path = os.path.join(CACHE_DIR, f"{sheet_name}.parquet")
    if not os.path.exists(cache_path):
        scope = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        if os.path.exists(JSON_KEY_FILE):
            creds = Credentials.from_service_account_file(JSON_KEY_FILE, scopes=scope)
        elif "gcp_service_account" in st.secrets:
            creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        else:
            st.error("Kredensial tidak ditemukan.")
            st.stop()
            
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SHEET_ID)
        worksheet = spreadsheet.worksheet(sheet_name)
        df = pd.DataFrame(worksheet.get_all_records())
        df = df.astype(str).replace(['nan', 'None', ''], 'Unknown').drop_duplicates()
        df.to_parquet(cache_path, engine='pyarrow', index=False)
        return df
    return pd.read_parquet(cache_path)

# --- 5. CORE PROCESSING ---
try:
    with st.spinner('Memuat Data & Filter Lengkap...'):
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

    df_hpp_clean = df_hpp_raw.drop_duplicates(subset=['ITEM_NO'], keep='last').copy()
    df_hpp_clean['ITEM_NO'] = df_hpp_clean['ITEM_NO'].astype(str)
    df_hpp_clean['HPP'] = pd.to_numeric(df_hpp_clean['HPP'], errors='coerce').fillna(0)

    # B. Intelligence Mapping
    df_sales['MY'] = df_sales['FORM_DATE'].dt.to_period('M')
    total_months = max(df_sales['MY'].nunique(), 1)
    
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

    df_master_clean = df_master_raw[['ITEM_NO', 'ITEM_NAME', 'GROUP_NAME1', 'GROUP_NAME2', 'GROUP_NAME3', 'GROUP_NAME4', 'VENDOR_NAME']].drop_duplicates(subset=['ITEM_NO'])
    df_intel = pd.merge(df_master_clean, cons[['ITEM_NO', 'KATEGORI']], on='ITEM_NO', how='left')
    df_intel = pd.merge(df_intel, avg_s[['ITEM_NO', 'FSD', 'QTY']], on='ITEM_NO', how='left')
    df_intel['AVG_SALES'] = pd.to_numeric(df_intel['QTY'], errors='coerce').fillna(0)

    df_final = pd.merge(df_sales, df_intel, on='ITEM_NO', how='left', suffixes=('', '_dup'))
    df_final = df_final.loc[:, ~df_final.columns.str.contains('_dup')]
    df_final = pd.merge(df_final, df_hpp_clean[['ITEM_NO', 'HPP']], on='ITEM_NO', how='left')
    df_final['PROFIT'] = df_final['NET_AMOUNT'] - (df_final['QTY'] * df_final['HPP'])

    # C. Stok Valuasi
    df_stok_val = pd.merge(df_stok_raw.assign(ITEM_NO=df_stok_raw['ITEM_NO'].astype(str)), df_intel, on='ITEM_NO', how='left')
    df_stok_val = pd.merge(df_stok_val, df_hpp_clean[['ITEM_NO', 'HPP']], on='ITEM_NO', how='left')
    df_stok_val['BALANCE_QTY'] = pd.to_numeric(df_stok_val['BALANCE_QTY'], errors='coerce').fillna(0)
    df_stok_val['STOCK_RUPIAH'] = df_stok_val['BALANCE_QTY'] * df_stok_val['HPP'].fillna(0)

    # --- 6. SIDEBAR FILTERS (FULL PENGEMBALIAN) ---
    st.sidebar.header("🔍 Global Filters")
    all_item_list = sorted([str(x) for x in df_intel['ITEM_NAME'].unique().tolist()])
    search_item = st.sidebar.selectbox("🔎 Cari Nama Barang", ["Semua Barang"] + all_item_list)

    def multi_filter(label, col, df):
        options = sorted([str(x) for x in df[col].unique().tolist() if str(x) != 'nan' and str(x) != 'Unknown'])
        return st.sidebar.multiselect(label, options)

    f_thrg = multi_filter("TIPE HARGA (Promo/Reguler)", 'TIPE_HARGA', df_final)
    f_type = multi_filter("FORM_TYPE", 'FORM_TYPE', df_final)
    f_tahun = multi_filter("TAHUN", 'TAHUN', df_final)
    f_bulan = multi_filter("BULAN", 'BULAN', df_final)
    f_vend = multi_filter("VENDOR NAME", 'VENDOR_NAME', df_final)
    
    st.sidebar.divider()
    f_kat = multi_filter("KATEGORI (Intel)", 'KATEGORI', df_final)
    f_fsd = multi_filter("FSD (Velocity)", 'FSD', df_final)
    
    st.sidebar.divider()
    f_g1 = multi_filter("GROUP 1 (Dept)", 'GROUP_NAME1', df_final)
    f_g2 = multi_filter("GROUP 2", 'GROUP_NAME2', df_final)
    f_g3 = multi_filter("GROUP 3", 'GROUP_NAME3', df_final)
    f_g4 = multi_filter("GROUP 4", 'GROUP_NAME4', df_final)

    # Filtering Logic Apply
    df_f = df_final.copy()
    if search_item != "Semua Barang": df_f = df_f[df_f['ITEM_NAME'] == search_item]
    for col, f in [('TIPE_HARGA', f_thrg), ('FORM_TYPE', f_type), ('TAHUN', f_tahun), 
                   ('BULAN', f_bulan), ('VENDOR_NAME', f_vend), ('KATEGORI', f_kat), 
                   ('FSD', f_fsd), ('GROUP_NAME1', f_g1), ('GROUP_NAME2', f_g2), 
                   ('GROUP_NAME3', f_g3), ('GROUP_NAME4', f_g4)]:
        if f: df_f = df_f[df_f[col].isin(f)]

    df_s_f = df_stok_val.copy()
    for col, f in [('VENDOR_NAME', f_vend), ('KATEGORI', f_kat), ('FSD', f_fsd), 
                   ('GROUP_NAME1', f_g1), ('GROUP_NAME2', f_g2), ('GROUP_NAME3', f_g3), ('GROUP_NAME4', f_g4)]:
        if f: df_s_f = df_s_f[df_s_f[col].isin(f)]

    # --- 7. MAIN UI ---
    st.title("📊 Grand Mitra Intelligence Dashboard")
    
    rev, prof = df_f['NET_AMOUNT'].sum(), df_f['PROFIT'].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"Rp {rev:,.0f}")
    c2.metric("Total Margin", f"Rp {prof:,.0f}")
    c3.metric("Margin (%)", f"{(prof/rev*100 if rev>0 else 0):.2f}%")
    c4.metric("Stock Value", f"Rp {df_s_f['STOCK_RUPIAH'].sum():,.0f}")

    st.divider()
    tabs = st.tabs(["📋 Resume Performa", "📈 Trend Penjualan", "📦 Daftar SOQ", "🧪 Bedah Korelasi Dept", "🏆 Top 5 Promo Items"])

    with tabs[0]: # RESUME
        df_res = df_f.groupby('GROUP_NAME1').agg({'NET_AMOUNT': 'sum', 'PROFIT': 'sum', 'QTY': 'sum', 'ITEM_NO': 'nunique'}).reset_index()
        st.dataframe(df_res.sort_values('NET_AMOUNT', ascending=False).style.format({'NET_AMOUNT': 'Rp {:,.0f}', 'PROFIT': 'Rp {:,.0f}'}), use_container_width=True)

    with tabs[2]: # SOQ
        df_soq = df_s_f.copy()
        df_soq['SOQ'] = (df_soq['AVG_SALES'] * 3) - df_soq['BALANCE_QTY']
        st.dataframe(df_soq[df_soq['SOQ'] > 0][['ITEM_NO', 'ITEM_NAME', 'BALANCE_QTY', 'SOQ']].sort_values('SOQ', ascending=False), use_container_width=True)

    with tabs[3]: # 5W+1H DINAMIS
        st.subheader("🧪 Analisis Kedalaman Efektivitas Promo")
        g_corr = df_f['IS_PROMO'].corr(df_f['NET_AMOUNT']) if df_f['IS_PROMO'].nunique() > 1 else 0
        g_avg_r = df_f[df_f['IS_PROMO'] == 0]['NET_AMOUNT'].mean()
        g_avg_p = df_f[df_f['IS_PROMO'] == 1]['NET_AMOUNT'].mean()
        g_lift = ((g_avg_p - g_avg_r) / g_avg_r * 100) if g_avg_r > 0 else 0

        if g_lift > 200 and g_corr < 0.2:
            s_what, s_why = "Audit Transaksi Proyek", f"Lompatan omzet sangat tinggi ({g_lift:.1f}%) tapi korelasi rendah. Promo dimanfaatkan transaksi besar."
        elif g_corr > 0.3:
            s_what, s_why = "Ekspansi Program Loyalitas", "Pelanggan merespon promo secara konsisten dan sehat."
        else:
            s_what, s_why = "Review Display & Item", "Dampak promo ada namun belum kuat menggerakkan nilai belanja massa."

        col_ins1, col_ins2 = st.columns(2)
        with col_ins1:
            st.info(f"**Analisa Global:** Pearson: **{g_corr:.4f}** | Lift: **{g_lift:.2f}%**")
        with col_ins2:
            st.table(pd.DataFrame({
                "Point": ["What", "Where", "Why", "Who", "Whom", "How"],
                "Strategic Action": [s_what, "Area Showroom", s_why, "Pak Sofyan (CM)", "Retail & B2B", "Cek ketersediaan Top 5 Item."]
            }))

    with tabs[4]: # TOP PROMO
        df_p = df_f[df_f['IS_PROMO'] == 1]
        if not df_p.empty:
            st.dataframe(df_p.groupby(['GROUP_NAME1', 'ITEM_NAME']).agg({'NET_AMOUNT': 'sum', 'QTY': 'sum'}).reset_index().sort_values('NET_AMOUNT', ascending=False).head(20), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
