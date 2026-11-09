export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: { DEFAULT: "#1a1a19", raised: "#232322", sunken: "#131312", line: "#34342f" },
        ink: { DEFAULT: "#f0efec", soft: "#c3c2b7", muted: "#8f8e85" },
        accent: "#3987e5",
      },
      fontFamily: { mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"] },
    },
  },
  plugins: [],
};
