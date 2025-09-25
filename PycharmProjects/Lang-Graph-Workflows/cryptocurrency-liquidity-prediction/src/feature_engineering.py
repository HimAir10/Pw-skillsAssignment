from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """Create new features and scale existing ones"""
    df['liquidity_ratio'] = df['24h_volume'] / df['mkt_cap']
    numerical_cols = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler
