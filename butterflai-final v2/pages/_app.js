import { SessionProvider } from "next-auth/react"
import Head from "next/head"
import "../styles/globals.css"

export default function App({ Component, pageProps: { session, ...pageProps } }) {
  return (
    <SessionProvider session={session}>
      <Head>
        <title>ButterflAI — Train Any Model. End to End.</title>
        <meta name="description" content="Type one sentence. ButterflAI finds the dataset, generates training code, auto-fixes consistency issues, trains on real GPUs, and delivers a working model with a live demo." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
      </Head>
      <Component {...pageProps} />
    </SessionProvider>
  )
}
