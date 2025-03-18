import express, {Application, Request, Response} from "express"
import "dotenv/config"

const app: Application = express();

const PORT = process.env.PORT || 7000;

app.get("/", (req: any, res: any) => {
    res.send("Hey It's working... 🙌");
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));