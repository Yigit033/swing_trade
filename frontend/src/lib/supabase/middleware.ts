import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

export async function updateSession(request: NextRequest) {
  if (!supabaseUrl || !supabaseAnonKey) {
    return NextResponse.next();
  }

  let response = NextResponse.next({ request });
  const supabase = createServerClient(supabaseUrl, supabaseAnonKey, {
    cookies: {
      getAll() {
        return request.cookies.getAll();
      },
      setAll(cookiesToSet) {
        cookiesToSet.forEach(({ name, value, options }) =>
          response.cookies.set(name, value, options)
        );
      },
    },
  });

  const {
    data: { user },
  } = await supabase.auth.getUser();

  const isLoginPage = request.nextUrl.pathname === "/login";
  const isAuthPage = request.nextUrl.pathname.startsWith("/auth/");

  // Allow /login even when logged in — user can sign out from there
  if (user && isAuthPage) {
    return NextResponse.redirect(new URL("/", request.url));
  }

  if (!user && !isLoginPage && !isAuthPage) {
    const protectedPaths = ["/", "/trades", "/performance", "/pending", "/scanner", "/lookup", "/charts", "/chat", "/backtest"];
    const isProtected = protectedPaths.some(
      (p) => p === request.nextUrl.pathname || (p !== "/" && request.nextUrl.pathname.startsWith(p))
    );
    if (isProtected) {
      return NextResponse.redirect(new URL("/login", request.url));
    }
  }

  return response;
}
