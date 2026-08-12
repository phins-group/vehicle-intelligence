import fs from "fs";
import path from "path";
import crypto from "crypto";

const sourceFolder = "/Users/duyhuynh/Downloads/CarTGMT";
const outputFolder = "./samples/images/plate";

fs.mkdirSync(outputFolder, { recursive: true });

function getUniqueName(ext) {
  while (true) {
    const name = `${crypto.randomUUID()}${ext.toLowerCase()}`;
    const outputPath = path.join(outputFolder, name);

    if (!fs.existsSync(outputPath)) {
      return name;
    }
  }
}

function copyAllFiles(folder) {
  const items = fs.readdirSync(folder, { withFileTypes: true });

  for (const item of items) {
    const fullPath = path.join(folder, item.name);

    if (item.isDirectory()) {
      copyAllFiles(fullPath);
      continue;
    }

    if (!item.isFile()) {
      continue;
    }

    const ext = path.extname(item.name);
    const newName = getUniqueName(ext);
    const outputPath = path.join(outputFolder, newName);

    fs.copyFileSync(fullPath, outputPath);

    console.log(`${fullPath} -> ${newName}`);
  }
}

copyAllFiles(sourceFolder);

console.log("DONE");