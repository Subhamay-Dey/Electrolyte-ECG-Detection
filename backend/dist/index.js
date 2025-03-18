import express from "express";
import "dotenv/config";
const app = express();
const PORT = process.env.PORT || 7000;
app.get("/", (req, res) => {
    res.send("Hey It's working... 🙌");
});
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
