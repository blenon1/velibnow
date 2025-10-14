class RebalancingPolicy:
    def suggest(self, df, threshold=0.7, k=10):
        at_risk = df[df["proba"] > threshold].sort_values("proba", ascending=False).head(k)
        return at_risk[["station_id", "name", "proba", "numbikesavailable", "numdocksavailable"]]
